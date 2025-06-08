import sys
import os

# Add the project root directory to sys.path
# os.path.dirname(__file__) gives the directory of conftest.py (i.e., tests/)
# os.path.join(..., '..') goes one level up to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

