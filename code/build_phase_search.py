"""Build original MCTS with isolated, non-consuming RNG serialization bindings."""
from pathlib import Path
import difflib
import hashlib
import json
import re
import shutil
import subprocess
import sys

CODE = Path(__file__).resolve().parent
BUILD = CODE / 'cpp/build_phase'


def main():
    import pybind11
    original = {str(p.relative_to(CODE/'cpp')): p.read_text()
                for p in (CODE/'cpp').rglob('*.hpp')
                if not any(x.startswith('build') for x in p.relative_to(CODE/'cpp').parts)}
    original['bindings.cpp'] = (CODE/'cpp/bindings.cpp').read_text()
    generated = dict(original)
    text = original['bindings.cpp'].replace('PYBIND11_MODULE(bindings, m)', 'PYBIND11_MODULE(bindings_phase, m)')
    text = '#include <sstream>\n#include <locale>\n' + text
    text, count = re.subn(r'(pybind11::class_<[^>]+>\s*\(m, "[^"]+")\)', r'\1, pybind11::module_local())', text)
    if count != 7:
        raise ValueError('Unexpected binding classes')
    anchor = '.def("set_seed",[](Solver_Wrapper &wrapper,unsigned int seed) {wrapper.solver->g_gen.seed(seed);});'
    assert text.count(anchor) == 1
    text = text.replace(anchor, '''.def("set_seed",[](Solver_Wrapper &w,unsigned int seed) {w.solver->g_gen.seed(seed);})
        .def("get_rng_state", [](Solver_Wrapper &w) {
            std::ostringstream s; s.imbue(std::locale::classic()); s << w.solver->g_gen; return s.str();
        })
        .def("set_rng_state", [](Solver_Wrapper &w, const std::string &state) {
            std::istringstream s(state); s.imbue(std::locale::classic()); auto gen=w.solver->g_gen;
            if (!(s >> gen)) throw std::invalid_argument("Invalid solver RNG state");
            s >> std::ws; if (!s.eof()) throw std::invalid_argument("Trailing solver RNG state");
            w.solver->g_gen=gen;
        });''')
    # Problem randomness is currently supplied by the solver engine. Preserve
    # its distribution and auxiliary engine as well, rather than assuming this.
    anchor = 'pybind11::class_<Problem_Wrapper> (m, "Problem_Wrapper", pybind11::module_local())'
    if anchor not in text:
        anchor = 'pybind11::class_<Problem_Wrapper>(m, "Problem_Wrapper", pybind11::module_local())'
    assert text.count(anchor) == 1
    text = text.replace(anchor, anchor + '''
        .def("get_rng_state", [](Problem_Wrapper &w) {
            std::ostringstream s; s.imbue(std::locale::classic()); s << w.problem->m_gen << "\\n" << w.problem->dist; return s.str();
        })
        .def("set_rng_state", [](Problem_Wrapper &w, const std::string &state) {
            std::istringstream s(state); s.imbue(std::locale::classic()); auto gen=w.problem->m_gen; auto dist=w.problem->dist;
            if (!(s >> gen >> dist)) throw std::invalid_argument("Invalid problem RNG state");
            s >> std::ws; if (!s.eof()) throw std::invalid_argument("Trailing problem RNG state");
            w.problem->m_gen=gen; w.problem->dist=dist;
        })''')
    generated['bindings.cpp'] = text
    generated['CMakeLists.txt'] = '''cmake_minimum_required(VERSION 3.12)
project(bindings_phase)
set(CMAKE_CXX_STANDARD 14)
find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)
pybind11_add_module(bindings_phase bindings.cpp)
target_include_directories(bindings_phase PRIVATE ${EIGEN3_INCLUDE_DIRS})
target_compile_options(bindings_phase PRIVATE -O2)
'''
    source = BUILD/'source'
    for name, content in generated.items():
        path = source/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    diff = ''.join(''.join(difflib.unified_diff(original[n].splitlines(True), generated[n].splitlines(True),
        fromfile='original/'+n, tofile='private/'+n)) for n in original if original[n] != generated[n])
    (BUILD/'rng_bindings.diff').write_text(diff)
    subprocess.run(['cmake', '-S', str(source), '-B', str(BUILD/'cmake'), '-DCMAKE_BUILD_TYPE=Release',
        '-DPYBIND11_LTO_CXX_FLAGS=', f'-DPYTHON_EXECUTABLE={sys.executable}', f'-Dpybind11_DIR={pybind11.get_cmake_dir()}'], check=True)
    subprocess.run(['cmake', '--build', str(BUILD/'cmake'), '-j', '2'], check=True)
    binary = next((BUILD/'cmake').glob('bindings_phase*.so'))
    temporary = BUILD/(binary.name+'.tmp')
    shutil.copyfile(binary, temporary)
    temporary.replace(BUILD/binary.name)
    digest = lambda t: hashlib.sha256(t.encode()).hexdigest()
    manifest = dict(original={n: digest(t) for n,t in original.items()}, generated={n: digest(t) for n,t in generated.items()},
        binary=hashlib.sha256((BUILD/binary.name).read_bytes()).hexdigest())
    (BUILD/'build_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('Private RNG-only extension built:', BUILD/binary.name)


if __name__ == '__main__':
    main()
