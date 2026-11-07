"""Build and atomically publish the optional local-protocol C++ extensions."""
from pathlib import Path
import os
import shutil
import subprocess
import sys
import pybind11


def main():
    root=Path(__file__).resolve().parent/'cpp'
    build=root/'build_local_rl'
    subprocess.run(['cmake','-S',str(root),'-B',str(build),f'-DPYTHON_EXECUTABLE={sys.executable}',
                    f'-Dpybind11_DIR={pybind11.get_cmake_dir()}','-DCMAKE_BUILD_TYPE=Release'],check=True)
    subprocess.run(['cmake','--build',str(build),'-j','2'],check=True)
    source=next(build.glob('bindings*.so'));destination=root/'build'/source.name
    destination.parent.mkdir(exist_ok=True)
    backup=destination.with_suffix(destination.suffix+'.pre_local_rl')
    if destination.exists() and not backup.exists():shutil.copy2(destination,backup)
    temporary=destination.with_suffix(destination.suffix+'.new')
    shutil.copy2(source,temporary);os.replace(temporary,destination)
    print(f'Published {destination}; prior binary retained at {backup}',flush=True)


if __name__=='__main__':main()
