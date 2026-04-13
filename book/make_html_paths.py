import shutil
import os

DEST_DIR = './_build/html/'

SOURCE_DIRS = (
    './demo/',
    './bench/',
)

IGNORE = (
    '*.py', 
    '*.ipynb', 
    '*.pdf', 
    '*.h5', 
    '*.xdmf', 
    '*.npy', 
    '*.npz',
    '*.txt',
)

if __name__ == "__main__":
    for s in SOURCE_DIRS:
        shutil.copytree(
            s,
            os.path.join(DEST_DIR, s),
            ignore=shutil.ignore_patterns(*IGNORE),
            dirs_exist_ok=True,   
        )