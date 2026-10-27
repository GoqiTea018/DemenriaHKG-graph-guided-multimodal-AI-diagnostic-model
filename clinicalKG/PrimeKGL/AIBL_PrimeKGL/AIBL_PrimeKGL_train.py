"""Command-line entry for AIBL三模态+PrimeKG.ipynb."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from notebook_runner import train_main

if __name__ == "__main__":
    train_main(Path(__file__).with_name('AIBL三模态+PrimeKG.ipynb'))
