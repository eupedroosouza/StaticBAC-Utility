import os
import sys

root = os.path.dirname(__file__)
sys.path.insert(0, str(os.path.join(str(root), "src")))

import utils

utils._root = root

from app import run

if __name__ == "__main__":
    run()
