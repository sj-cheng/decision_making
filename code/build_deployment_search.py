"""Copy the frozen audit extension; add an isolated synchronous search baseline."""
from pathlib import Path
import difflib
import hashlib
import json
import shutil
import subprocess
import sys

CODE = Path(__file__).resolve().parent
BUILD = CODE/'cpp/build_deployment'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Unexpected source anchor: '+old[:90])
    return text.replace(old, new)


def main():
    import pybind11
    source_base=CODE/'cpp/build_root_audit/source'
    manifest=json.loads((CODE/'cpp/build_root_audit/build_manifest.json').read_text())
    for section,base in [('original',CODE/'cpp'),('generated',source_base)]:
        for name,value in manifest[section].items():
            if digest(base/name)!=value:
                raise ValueError('Frozen dependency changed: '+str(base/name))
    original={name:(source_base/name).read_text() for name in manifest['generated']}
    generated=dict(original)
    generated['bindings.cpp']=generated['bindings.cpp'].replace('bindings_root_audit','bindings_deployment')
    anchor='.def_readonly("root_audit", &Solver_Result::root_audit);'
    generated['bindings.cpp']=replace_once(generated['bindings.cpp'],anchor,
        anchor[:-1]+ '\n        .def_readonly("step_calls", &Solver_Result::step_calls)'
        '\n        .def_readonly("policy_evaluations", &Solver_Result::policy_evaluations)'
        '\n        .def_readonly("value_evaluations", &Solver_Result::value_evaluations);')
    generated['solvers/solver.hpp']=replace_once(generated['solvers/solver.hpp'],
        'Eigen::MatrixXf root_audit;', 'Eigen::MatrixXf root_audit;\n'
        '        long long step_calls=0, policy_evaluations=0, value_evaluations=0;')
    generated['problems/problem.hpp']=replace_once(generated['problems/problem.hpp'],
        '    int m_state_dim;', '    long long diagnostic_step_calls=0;\n    int m_state_dim;')
    generated['problems/example8.hpp']=replace_once(generated['problems/example8.hpp'],
        'Eigen::Matrix<float,-1,1> next_state = state;',
        '++diagnostic_step_calls;\n            Eigen::Matrix<float,-1,1> next_state = state;')
    search=generated['solvers/puct_v1.hpp']
    search=replace_once(search,'solver_result.success = false;\n\t\t\tm_two_teams',
        'solver_result.success = false;\n            const auto steps_before=problem->diagnostic_step_calls;\n'
        '            long long policies=0,values=0;\n\t\t\tm_two_teams')
    search=search.replace('auto encoding = problem->policy_encoding(state_of(parent), ii);',
        '++policies_counter;\n                        auto encoding = problem->policy_encoding(state_of(parent), ii);')
    # Expansion/default-policy helpers share counters; no random operations added.
    search=search.replace('long long policies=0,values=0;', 'policies_counter=values_counter=0;')
    search=replace_once(search,'auto encoding = problem->value_encoding(curr_state);',
        '++values_counter;\n                auto encoding = problem->value_encoding(curr_state);')
    search=replace_once(search,'bool m_export_root = false;',
        'long long policies_counter=0, values_counter=0;\n        bool m_export_root = false;')
    search=replace_once(search,'solver_result.num_visits = m_meta[0].num_visits;',
        'solver_result.num_visits = m_meta[0].num_visits;\n'
        '            solver_result.step_calls=problem->diagnostic_step_calls-steps_before;\n'
        '            solver_result.policy_evaluations=policies_counter;\n'
        '            solver_result.value_evaluations=values_counter;')
    generated['solvers/puct_v1.hpp']=search
    generated['solvers/simultaneous_puct.hpp']=(CODE/'cpp/solvers/simultaneous_puct.hpp').read_text()
    generated['solvers/solver_wrapper.hpp']=replace_once(generated['solvers/solver_wrapper.hpp'],
        '#include "puct_v1.hpp"', '#include "puct_v1.hpp"\n#include "simultaneous_puct.hpp"')
    generated['solvers/solver_wrapper.hpp']=replace_once(generated['solvers/solver_wrapper.hpp'],
        'if (string == "C_PUCT_V0"){', 'if (string == "C_PUCT_SYNC") {\n'
        '                solver = new SimultaneousPUCT();\n            } else if (string == "C_PUCT_V0"){')
    generated['CMakeLists.txt']=generated['CMakeLists.txt'].replace('bindings_root_audit','bindings_deployment')
    for name,content in generated.items():
        path=BUILD/'source'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(content)
    (BUILD/'changes.diff').write_text(''.join(''.join(difflib.unified_diff(original[n].splitlines(True),
        generated[n].splitlines(True),fromfile='frozen/'+n,tofile='private/'+n))
        for n in original if original[n]!=generated[n]))
    subprocess.run(['cmake','-S',str(BUILD/'source'),'-B',str(BUILD/'cmake'),
        '-DCMAKE_BUILD_TYPE=Release',f'-DPYTHON_EXECUTABLE={sys.executable}',
        f'-Dpybind11_DIR={pybind11.get_cmake_dir()}'],check=True)
    subprocess.run(['cmake','--build',str(BUILD/'cmake'),'-j','2'],check=True)
    binary=next((BUILD/'cmake').glob('bindings_deployment*.so'))
    shutil.copyfile(binary,BUILD/binary.name)
    record=dict(dependency_manifest=digest(CODE/'cpp/build_root_audit/build_manifest.json'),
        generated={n:digest(BUILD/'source'/n) for n in generated},
        binary=digest(BUILD/binary.name),builder=digest(Path(__file__)),
        synchronous_source=digest(CODE/'cpp/solvers/simultaneous_puct.hpp'))
    (BUILD/'build_manifest.json').write_text(json.dumps(record,indent=2)+'\n')
    print(BUILD/binary.name)


if __name__=='__main__':
    main()
