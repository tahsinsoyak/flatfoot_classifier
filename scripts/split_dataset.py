"""CLI script to generate stratified train, validation, and test splits."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.split import create_stratified_splits

if __name__ == "__main__":
    create_stratified_splits()
