"""Build the isolated C++ search used by memory_observation_v4 training.

The shared cpp/ sources stay untouched because several finished experiments
hash them.  This copies them, applies the V4 reward and rollout changes, and
builds a separate ``bindings_v4`` extension.
"""
import json
from pathlib import Path
import shutil
import subprocess
import sys

from build_deployment_search import digest, replace_once

CODE = Path(__file__).resolve().parent
SOURCE = CODE / 'cpp'
BUILD = CODE / 'cpp/build_v4'

# Must match problems/example8_v4.py.
EVADER_SURVIVAL_REWARD = 0.005
GREEDY_ROLLOUT_PROB = 0.8


def copy_sources():
    files = {}
    for path in sorted(SOURCE.rglob('*')):
        rel = path.relative_to(SOURCE)
        if not path.is_file() or rel.parts[0].startswith('build') or rel.parts[0] == 'testing':
            continue
        if path.suffix not in ('.hpp', '.h', '.cpp', '.txt'):
            continue
        files[str(rel)] = path.read_text()
    return files


def patch_example8(text):
    text = replace_once(
        text,
        'r1=-0.5f*newly_captured*(1.0f - t / m_tf);',
        'r1=-0.5f*newly_captured;',
    )
    text = replace_once(
        text,
        '\t\t\tadd_evader_distance_reward(r, next_state);\n',
        '\t\t\tadd_evader_distance_reward(r, next_state);\n'
        '\t\t\t{\n'
        '\t\t\t\tconst int alive = active_evader_count(next_state);\n'
        '\t\t\t\tfor (int e : m_evaders) {\n'
        '\t\t\t\t\tr(e,0) += {w}f * alive;\n'
        '\t\t\t\t}\n'
        '\t\t\t}\n'.replace('{w}', repr(EVADER_SURVIVAL_REWARD)),
    )
    text = replace_once(
        text,
        '\t\t\t\tEigen::Matrix<float,-1,1> probe = state;\n'
        '\t\t\t\tbool found = false;\n',
        '\t\t\t\tEigen::Matrix<float,-1,1> probe = state;\n'
		'\t\t\t\tbool found = false;\n'
		'\t\t\t\t// Random rollouts never close a corner, so they rated corners as safe.\n'
		'\t\t\t\t// Pursuers chase and evaders flee the nearest opponent at full speed;\n'
		'\t\t\t\t// a flee blocked by a wall falls back to the random legal step below.\n'
		'\t\t\t\tconst bool is_pursuer =\n'
		'\t\t\t\t\tstd::find(m_pursuers.begin(), m_pursuers.end(), robot) != m_pursuers.end();\n'
		'\t\t\t\tif (dist(gen) < {p}) {\n'
		'\t\t\t\t\tfloat nearest = std::numeric_limits<float>::infinity();\n'
		'\t\t\t\t\tVec2f target = pos;\n'
		'\t\t\t\t\tfor (int other : (is_pursuer ? m_evaders : m_pursuers)) {\n'
		'\t\t\t\t\t\tif (!is_active(state, other)) continue;\n'
		'\t\t\t\t\t\tconst Vec2f other_pos = state.block(m_state_idxs[other][0], 0, 2, 1);\n'
		'\t\t\t\t\t\tconst float d = (other_pos - pos).norm();\n'
		'\t\t\t\t\t\tif (d < nearest) {\n'
		'\t\t\t\t\t\t\tnearest = d;\n'
		'\t\t\t\t\t\t\ttarget = other_pos;\n'
		'\t\t\t\t\t\t}\n'
		'\t\t\t\t\t}\n'
		'\t\t\t\t\tif (std::isfinite(nearest) && nearest > 1e-6f) {\n'
		'\t\t\t\t\t\tVec2f delta = target - pos;\n'
		'\t\t\t\t\t\tif (!is_pursuer) delta *= -1e3f;\n'
        '\t\t\t\t\t\tfloat scale = 1.0f;\n'
        '\t\t\t\t\t\tfor (int ii = 0; ii < 2; ++ii) {\n'
        '\t\t\t\t\t\t\tif (delta(ii) > bounds.second(ii)) scale = std::min(scale, bounds.second(ii) / delta(ii));\n'
        '\t\t\t\t\t\t\tif (delta(ii) < bounds.first(ii)) scale = std::min(scale, bounds.first(ii) / delta(ii));\n'
        '\t\t\t\t\t\t}\n'
        '\t\t\t\t\t\tconst Vec2f a = scale * delta;\n'
        '\t\t\t\t\t\tprobe.block(m_state_idxs[robot][0], 0, 2, 1) = pos + a;\n'
        '\t\t\t\t\t\tif (is_robot_state_valid(probe, robot)) {\n'
        '\t\t\t\t\t\t\taction.block(a0, 0, 2, 1) = a;\n'
        '\t\t\t\t\t\t\tfound = true;\n'
        '\t\t\t\t\t\t}\n'
        '\t\t\t\t\t}\n'
		'\t\t\t\t}\n'.replace('{p}', repr(GREEDY_ROLLOUT_PROB)),
    )
    return text


def main():
    import pybind11
    source = copy_sources()
    source['problems/example8.hpp'] = patch_example8(source['problems/example8.hpp'])
    source['bindings.cpp'] = replace_once(
        source['bindings.cpp'], 'PYBIND11_MODULE(bindings, m)', 'PYBIND11_MODULE(bindings_v4, m)')
    source['CMakeLists.txt'] = replace_once(
        source['CMakeLists.txt'], 'pybind11_add_module(bindings bindings.cpp)',
        'pybind11_add_module(bindings_v4 bindings.cpp)')
    source['CMakeLists.txt'] = replace_once(
        source['CMakeLists.txt'], 'set(CMAKE_BUILD_TYPE Debug)', 'set(CMAKE_BUILD_TYPE Release)')

    if (BUILD / 'source').exists():
        shutil.rmtree(BUILD / 'source')
    for name, content in source.items():
        path = BUILD / 'source' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    subprocess.run(['cmake', '-S', str(BUILD / 'source'), '-B', str(BUILD / 'cmake'),
        '-DCMAKE_BUILD_TYPE=Release', f'-DPYTHON_EXECUTABLE={sys.executable}',
        f'-Dpybind11_DIR={pybind11.get_cmake_dir()}'], check=True)
    subprocess.run(['cmake', '--build', str(BUILD / 'cmake'), '-j', '8'], check=True)
    binary = next((BUILD / 'cmake').glob('bindings_v4*.so'))
    staged = BUILD / (binary.name + '.tmp')
    shutil.copyfile(binary, staged)
    staged.replace(BUILD / binary.name)
    (BUILD / 'build_manifest.json').write_text(json.dumps(dict(
        evader_survival_reward=EVADER_SURVIVAL_REWARD,
		greedy_rollout_prob=GREEDY_ROLLOUT_PROB,
        upstream={name: digest(SOURCE / name) for name in source},
        files={name: digest(BUILD / 'source' / name) for name in source},
        binary=digest(BUILD / binary.name),
        builder=digest(Path(__file__)),
    ), indent=2) + '\n')


if __name__ == '__main__':
    main()
