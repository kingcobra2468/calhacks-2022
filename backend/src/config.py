import os
ROOT_DIR = None

if not ROOT_DIR:
    ROOT_DIR = os.path.abspath(os.curdir)
    print('ROOT DIR ', ROOT_DIR)