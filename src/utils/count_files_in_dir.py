"""Script to count the number of files in each subdirectory of a given parent directory."""
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.config import PROJECT_ROOT

parent_directory = f"{PROJECT_ROOT}/data/raw/food-101"

for root, dirs, files in os.walk(parent_directory):
    # 'root' is the current directory being walked
    # 'files' is a list of files in the current 'root' directory
    print(f"Files in {root}: {len(files)}")