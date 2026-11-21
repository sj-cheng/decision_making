"""Keep experiment scripts importable by their original module names."""
from pathlib import Path
import sys

CODE = Path(__file__).resolve().parent
SERIES = ('benchmarks', 'e44', 'e45', 'e46', 'e48', 'e49', 'e50', 'e51', 'e52', 'e55', 'l45', 'l5', 'large_pursuit_exp', 'search_rl', 'v3')

def install():
    entries = [CODE]
    for name in SERIES:
        folder = CODE / name
        if folder.is_dir():
            entries.append(folder)
    for folder in entries:
        text = str(folder)
        if text not in sys.path:
            sys.path.insert(1, text)

install()
