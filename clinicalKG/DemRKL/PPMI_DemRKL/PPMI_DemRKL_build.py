"""Command-line entry for PPMI_DemRKL_build.ipynb."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from notebook_runner import build_main

if __name__ == "__main__":
    build_main(Path(__file__).with_name('PPMI_DemRKL_build.ipynb'), 'DemRKL')
