import os
import logging
import platform
from pathlib import Path

# path in current working directory of the paths.py file
path = Path(os.path.realpath(__file__))

folder_dir: str = path.parents[1]
result_dir: str = os.path.join(folder_dir, "results")
