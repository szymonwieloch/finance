'''
Because Python and Marimo do not support imports of files located in different directories,
this file sets needed import paths.
'''

import sys
from os import path
parent_dir = path.dirname(path.dirname(__file__))

if parent_dir not in sys.path:
    sys.path.append(parent_dir)