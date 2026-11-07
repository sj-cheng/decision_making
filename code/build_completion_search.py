"""Build a private, reproducible MCTS completion-utility extension.

Historical sources and shared libraries remain byte-for-byte unchanged.
The generated source and unified diff are retained beside the private library.
"""
from pathlib import Path
import difflib
import hashlib
import json
import shutil
import subprocess
import sys

CODE = Path(__file__).resolve().parent
BUILD = CODE / 'cpp/build_completion'


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise RuntimeError(f'Historical source anchor changed: {before[:100]!r}')
    return text.replace(before, after)


def generated_sources():
    original = {str(p.relative_to(CODE/'cpp')): p.read_text()
                for p in (CODE/'cpp').rglob('*.hpp') if not any(
                    part.startswith('build') for part in p.relative_to(CODE/'cpp').parts)}
    original['bindings.cpp'] = (CODE/'cpp/bindings.cpp').read_text()
    generated = dict(original)
    path = 'solvers/solver.hpp'
    text = replace_once(original[path], 'bool export_tree = false;',
                        'bool export_tree = false;\n        float completion_bonus = 0.f;')
    text = replace_once(text, 'bool success;', '''bool success;
        long completion_tree_events = 0;
        long completion_rollout_events = 0;''')
    text = replace_once(text, 'virtual ~Solver() { }', '''virtual ~Solver() { }
        float completion_bonus = 0.f;
        long completion_tree_events = 0;
        long completion_rollout_events = 0;

        Eigen::VectorXf transition_utility(Problem* problem,
            const Eigen::VectorXf& state, const Eigen::VectorXf& action,
            const Eigen::VectorXf& next, bool rollout) {
            Eigen::VectorXf reward = problem->normalized_reward_from_next(state, action, next);
            // Example8 is the only supported completion task. The guard also
            // excludes already-terminal, illegal, first-capture and timeout states.
            if (completion_bonus != 0.f && problem->m_num_robots == 4 &&
                problem->m_state_dim == 53 && !problem->is_terminal(state) &&
                (problem->is_active(state, 0) || problem->is_active(state, 1)) &&
                !problem->is_active(next, 0) && !problem->is_active(next, 1) &&
                problem->is_valid(next)) {
                reward(2) += completion_bonus;
                reward(3) += completion_bonus;
                if (rollout) ++completion_rollout_events;
                else ++completion_tree_events;
            }
            return reward;
        }''')
    generated[path] = text
    path = 'solvers/puct_v1.hpp'
    text = replace_once(original[path], 'm_export_tree = solver_settings.export_tree;',
                        'm_export_tree = solver_settings.export_tree;\n            completion_bonus = solver_settings.completion_bonus;')
    text = replace_once(text, 'solver_result.success = false;\n\t\t\tm_two_teams',
                        'solver_result.success = false;\n            completion_tree_events = completion_rollout_events = 0;\n\t\t\tm_two_teams')
    text = replace_once(text, 'auto edge_reward = problem->normalized_reward_from_next(\n\t\t\t\tstate_of(parent), action, next_state);',
                        'auto edge_reward = transition_utility(problem, state_of(parent), action, next_state, false);')
    text = replace_once(text, 'problem->normalized_reward_from_next(curr_state, action, next_state);',
                        'transition_utility(problem, curr_state, action, next_state, true);')
    text = replace_once(text, 'solver_result.num_visits = m_meta[0].num_visits;',
                        'solver_result.num_visits = m_meta[0].num_visits;\n            solver_result.completion_tree_events = completion_tree_events;\n            solver_result.completion_rollout_events = completion_rollout_events;')
    generated[path] = text
    path = 'bindings.cpp'
    text = original[path].replace('PYBIND11_MODULE(bindings, m)', 'PYBIND11_MODULE(bindings_completion, m)')
    import re
    text, count = re.subn(r'(pybind11::class_<[^>]+>\s*\(m, "[^"]+")\)',
                          r'\1, pybind11::module_local())', text)
    if count != 7:
        raise RuntimeError('Private pybind class isolation changed')
    text = replace_once(text, '.def_readwrite("export_tree", &Solver_Settings::export_tree);',
                        '.def_readwrite("export_tree", &Solver_Settings::export_tree)\n        .def_readwrite("completion_bonus", &Solver_Settings::completion_bonus);')
    text = replace_once(text, '.def_readwrite("success", &Solver_Result::success);',
                        '.def_readwrite("success", &Solver_Result::success)\n        .def_readonly("num_visits", &Solver_Result::num_visits)\n        .def_readonly("completion_tree_events", &Solver_Result::completion_tree_events)\n        .def_readonly("completion_rollout_events", &Solver_Result::completion_rollout_events);')
    text = replace_once(text, '.def("set_seed",[](Solver_Wrapper &wrapper,unsigned int seed) {wrapper.solver->g_gen.seed(seed);});',
                        '''.def("set_seed",[](Solver_Wrapper &wrapper,unsigned int seed) {wrapper.solver->g_gen.seed(seed);})
        .def_property_readonly("completion_bonus", [](Solver_Wrapper& w) {return w.solver->completion_bonus;})
        .def("transition_utility", [](Solver_Wrapper& w, Problem_Wrapper& p,
                Eigen::VectorXf state, Eigen::VectorXf action, Eigen::VectorXf next, bool rollout) {
            return w.solver->transition_utility(p.problem, state, action, next, rollout);
        });''')
    generated[path] = text
    generated['CMakeLists.txt'] = '''cmake_minimum_required(VERSION 3.12)
project(bindings_completion)
set(CMAKE_CXX_STANDARD 14)
find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)
pybind11_add_module(bindings_completion bindings.cpp)
target_include_directories(bindings_completion PRIVATE ${EIGEN3_INCLUDE_DIRS})
target_compile_options(bindings_completion PRIVATE -O2)
'''
    return original, generated


def main():
    import pybind11
    original, generated = generated_sources()
    source = BUILD/'source'
    for name, text in generated.items():
        path = source/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    diff = ''.join(''.join(difflib.unified_diff(original[name].splitlines(True),
                    generated[name].splitlines(True), fromfile='original/'+name,
                    tofile='private/'+name)) for name in original if original[name] != generated[name])
    (BUILD/'completion_utility.diff').write_text(diff)
    subprocess.run(['cmake', '-S', str(source), '-B', str(BUILD/'cmake'),
                    '-DCMAKE_BUILD_TYPE=Release', '-DPYBIND11_LTO_CXX_FLAGS=',
                    f'-DPYTHON_EXECUTABLE={sys.executable}', f'-Dpybind11_DIR={pybind11.get_cmake_dir()}'], check=True)
    subprocess.run(['cmake', '--build', str(BUILD/'cmake'), '-j', '2'], check=True)
    binary = next((BUILD/'cmake').glob('bindings_completion*.so'))
    temporary = BUILD/(binary.name+'.tmp')
    shutil.copyfile(binary, temporary)
    temporary.replace(BUILD/binary.name)
    digest = lambda text: hashlib.sha256(text.encode()).hexdigest()
    manifest = dict(original={name: digest(text) for name, text in original.items()},
                    generated={name: digest(text) for name, text in generated.items()},
                    binary=hashlib.sha256((BUILD/binary.name).read_bytes()).hexdigest())
    (BUILD/'build_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('Private completion solver built:', BUILD/binary.name)


if __name__ == '__main__':
    main()
