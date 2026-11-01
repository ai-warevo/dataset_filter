import sys
from pathlib import Path

# Resolving the 'src/' layout boundary for direct script execution
SRC_PATH = str(Path(__file__).resolve().parent / "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from dataset_filter.application.bootstrap import Bootstrap

if __name__ == "__main__":
    Bootstrap().run()
