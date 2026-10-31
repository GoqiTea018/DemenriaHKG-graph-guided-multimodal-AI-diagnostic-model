"""Train DistMult once on the complete kg.csv, with safe stop and exact resume.

Run: python -m clinicalKG.PrimeKGL.full_primekg --epochs 50
Resume: python -m clinicalKG.PrimeKGL.full_primekg --epochs 50 --resume
No patient tables, seed lists, blacklist filtering, or hop extraction are used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import threading

import torch
from torch import nn
import numpy as np
import pandas as pd

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from clinicalKG.run.common.paths import SHARED_PRIMEKG, raw_primekg_path


class DistMult(nn.Module):
    """Original score, Xavier initialization, and margin-ranking objective."""
    def __init__(self, num_entities, num_relations, embed_dim):
        super().__init__()
        self.num_entities = num_entities
        self.ent_emb = nn.Embedding(num_entities, embed_dim)
        self.rel_emb = nn.Embedding(num_relations, embed_dim)
        nn.init.xavier_uniform_(self.ent_emb.weight)
        nn.init.xavier_uniform_(self.rel_emb.weight)
        self.criterion = nn.MarginRankingLoss(margin=1.0)

    def forward(self, h, r, t):
        return torch.sum(self.ent_emb(h) * self.rel_emb(r) * self.ent_emb(t), dim=1)

    def calculate_loss(self, h, r, t):
        neg_t = torch.randint(0, self.num_entities, (h.size(0),), device=h.device)
        return self.criterion(self(h, r, t), self(h, r, neg_t),
                              torch.ones(h.size(0), device=h.device))


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def load_graph(path):
    """Keep every original edge record, using the original name-based indexing."""
    path = Path(path)
    columns = pd.read_csv(path, nrows=0).columns
    if {"x_name", "relation", "y_name"}.issubset(columns):
        cols = ["x_name", "relation", "y_name"]
    elif {"head", "relation", "tail"}.issubset(columns):
        cols = ["head", "relation", "tail"]
    else:
        raise ValueError("Graph needs x_name/relation/y_name or head/relation/tail")
    print(f"Reading complete graph: {path}", flush=True)
    frame = pd.read_csv(path, usecols=cols, dtype=str)[cols]
    if frame.empty or frame.isna().any().any():
        raise ValueError("Graph is empty or contains missing entity/relation names")
    entities = sorted(set(frame[cols[0]]) | set(frame[cols[2]]))
    relations = sorted(set(frame[cols[1]]))
    ent2id = {name: index for index, name in enumerate(entities)}
    rel2id = {name: index for index, name in enumerate(relations)}
    encoded = np.column_stack((frame[cols[0]].map(ent2id).to_numpy(dtype=np.int64),
                               frame[cols[1]].map(rel2id).to_numpy(dtype=np.int64),
                               frame[cols[2]].map(ent2id).to_numpy(dtype=np.int64)))
    del frame
    return torch.from_numpy(encoded), ent2id, rel2id


def train_full_primekg(raw_kg=None, output_dir=None, *, epochs=50, batch_size=512,
                       embed_dim=128, lr=0.005, device=None, seed=2026,
                       resume=False, save_every_steps=1000, log_every_steps=100):
    """SIGINT/SIGTERM request a save after the current optimizer step finishes."""
    if min(epochs, batch_size, embed_dim, save_every_steps, log_every_steps) <= 0 or lr <= 0:
        raise ValueError("Training parameters must be positive")
    raw_kg = Path(raw_kg) if raw_kg is not None else raw_primekg_path()
    output = Path(output_dir) if output_dir is not None else SHARED_PRIMEKG
    checkpoint = output / "PrimeKG-DistMult-last.pt"
    if checkpoint.exists() and not resume:
        raise FileExistsError(f"Checkpoint exists: {checkpoint}. Use --resume or a new --output-dir.")
    if resume and not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    triplets, ent2id, rel2id = load_graph(raw_kg)
    fingerprint = _sha256(raw_kg)
    print(f"Full PrimeKG: {len(triplets)} records, {len(ent2id)} named entities, "
          f"{len(rel2id)} relations", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = DistMult(len(ent2id), len(rel2id), embed_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # The next unprocessed row in this epoch's saved permutation.
    epoch, next_row, global_step = 0, 0, 0
    permutation, partial_loss, partial_batches, losses = None, 0.0, 0, []
    config = {"embed_dim": embed_dim, "batch_size": batch_size, "lr": lr,
              "seed": seed, "graph_sha256": fingerprint, "records": len(triplets),
              "entities": len(ent2id), "relations": len(rel2id)}
    if resume:
        # Only load checkpoints generated by this script / trusted training jobs.
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if state["config"] != config:
            raise ValueError("Checkpoint graph/settings differ; keep the original graph and training settings")
        if state["entity2id"] != ent2id or state["relation2id"] != rel2id:
            raise ValueError("Checkpoint entity/relation indices differ")
        if state["epoch"] > epochs:
            raise ValueError("--epochs is the total target and is below completed epochs")
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        epoch, next_row, global_step = state["epoch"], state["next_row"], state["global_step"]
        permutation = state["permutation"]
        partial_loss, partial_batches = state["partial_loss"], state["partial_batches"]
        losses = state["loss_history"]
        random.setstate(state["python_rng_state"])
        np.random.set_state(state["numpy_rng_state"])
        torch.set_rng_state(state["torch_rng_state"])
        if device.type == "cuda" and state["cuda_rng_state"] is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng_state"])
        print(f"Resuming epoch {epoch + 1}, next row {next_row}, step {global_step}", flush=True)

    requested_stop = False

    def request_stop(signum, frame):
        nonlocal requested_stop
        requested_stop = True

    def save(reason):
        # CUDA operations complete before serialization; checkpoint is atomically replaced.
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        state = {"format_version": 1, "config": config, "epoch": epoch,
                 "next_row": next_row, "global_step": global_step,
                 "permutation": permutation, "partial_loss": partial_loss,
                 "partial_batches": partial_batches, "loss_history": losses,
                 "model_state_dict": model.state_dict(),
                 "optimizer_state_dict": optimizer.state_dict(),
                 "entity2id": ent2id, "relation2id": rel2id,
                 "python_rng_state": random.getstate(), "numpy_rng_state": np.random.get_state(),
                 "torch_rng_state": torch.get_rng_state(),
                 "cuda_rng_state": torch.cuda.get_rng_state_all() if device.type == "cuda" else None}
        tmp = checkpoint.with_name(checkpoint.name + ".tmp")
        torch.save(state, tmp)
        os.replace(tmp, checkpoint)
        _atomic_json(output / "PrimeKG-Entity2ID.json", ent2id)
        _atomic_json(output / "PrimeKG-Relation2ID.json", rel2id)
        _atomic_json(output / "PrimeKG-LossHistory.json", losses)
        embedding_path = output / "PrimeKG-Embeddings.npy"
        with embedding_path.with_name(embedding_path.name + ".tmp").open("wb") as handle:
            np.save(handle, model.ent_emb.weight.detach().cpu().numpy())
        os.replace(embedding_path.with_name(embedding_path.name + ".tmp"), embedding_path)
        _atomic_json(output / "PrimeKG-TrainingState.json",
                     {"reason": reason, "completed_epochs": epoch, "next_row": next_row,
                      "global_step": global_step, "partial_epoch_batches": partial_batches,
                      "raw_kg": str(raw_kg.resolve()), **config})
        print(f"Saved ({reason}): {checkpoint}; completed epochs={epoch}, step={global_step}", flush=True)

    previous_handlers = {}
    if threading.current_thread() is threading.main_thread():
        for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
            sig = getattr(signal, name, None)
            if sig is not None:
                previous_handlers[sig] = signal.signal(sig, request_stop)
    try:
        save("start" if not resume else "resume")
        while epoch < epochs and not requested_stop:
            if permutation is None:
                permutation = torch.randperm(len(triplets))
            model.train()
            while next_row < len(triplets) and not requested_stop:
                end = min(next_row + batch_size, len(triplets))
                batch = triplets[permutation[next_row:end]].to(device)
                optimizer.zero_grad()
                loss = model.calculate_loss(batch[:, 0], batch[:, 1], batch[:, 2])
                loss.backward()
                optimizer.step()
                batch_loss = float(loss.detach())
                partial_loss += batch_loss
                partial_batches += 1
                global_step += 1
                next_row = end
                if (partial_batches == 1 or global_step % log_every_steps == 0
                        or next_row == len(triplets) or requested_stop):
                    total_batches = (len(triplets) + batch_size - 1) // batch_size
                    print(f"Epoch [{epoch + 1:03d}/{epochs}] | "
                          f"Batch [{partial_batches}/{total_batches}] | Step {global_step} | "
                          f"Loss: {batch_loss:.4f} | "
                          f"Epoch Mean Loss: {partial_loss / partial_batches:.4f}", flush=True)
                if global_step % save_every_steps == 0:
                    save("periodic")
            if next_row == len(triplets):
                losses.append(partial_loss / partial_batches)
                epoch += 1
                next_row, permutation, partial_loss, partial_batches = 0, None, 0.0, 0
                print(f"Epoch [{epoch:03d}/{epochs}] | Train Loss: {losses[-1]:.4f}", flush=True)
                save("epoch")
        save("interrupted" if requested_stop else "complete")
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
    return {"checkpoint": checkpoint, "interrupted": requested_stop,
            "completed_epochs": epoch, "global_step": global_step}


def plot_loss(output_dir=None):
    import matplotlib.pyplot as plt
    output = Path(output_dir) if output_dir is not None else SHARED_PRIMEKG
    losses = json.loads((output / "PrimeKG-LossHistory.json").read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, len(losses) + 1), losses, label="DistMult Train Loss")
    ax.set(xlabel="Epochs", ylabel="Margin Ranking Loss", title="Full PrimeKG embedding loss")
    ax.grid(True, alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "PrimeKG-LossCurve.png", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-kg", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=50, help="Total target epochs, including completed epochs")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-every-steps", type=int, default=1000)
    parser.add_argument("--log-every-steps", type=int, default=100)
    args = parser.parse_args()
    result = train_full_primekg(**vars(args))
    plot_loss(args.output_dir)
    if result["interrupted"]:
        print("Stopped safely. Run the same command with --resume to continue.")


if __name__ == "__main__":
    main()
