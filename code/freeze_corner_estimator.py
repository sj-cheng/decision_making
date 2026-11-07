"""Freeze the accepted *research* estimator without changing deployment."""
from pathlib import Path
import json
import hashlib

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'current/l45_corner_turn_estimator_20261025'
PACKAGE='l45_corner_shared_20261025'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    result=json.loads((OUT/'confirmation_results.json').read_text())
    if not result['passed']:raise RuntimeError('Research estimator acceptance gate failed')
    protocol=json.loads((OUT/'experiment.json').read_text())
    for p,h in protocol['sources'].items():
        if sha(Path(p))!=h:raise RuntimeError('Accepted source changed')
    frozen=OUT/'frozen_estimator'
    if (frozen/'manifest.json').exists():raise RuntimeError('Accepted snapshot already exists; refusing overwrite')
    runtime=frozen/'runtime'/PACKAGE;runtime.mkdir(parents=True,exist_ok=True)
    (runtime/'__init__.py').write_text('"""Frozen local wall-turn estimator package; research only."""\n')
    lineage={}
    for name in ('corner_estimator.py','multimodal_estimator.py','history_recovery.py','local_pursuit.py'):
        src=ROOT/'code/solvers'/name;dest=runtime/name
        dest.write_text(src.read_text().replace('from solvers.','from '+PACKAGE+'.'))
        lineage[str(dest.relative_to(frozen))]=dict(source=str(src),original_sha256=sha(src),
            transformation='isolate import namespace only; numerical source unchanged')
    loader='''"""Load the frozen accepted estimator with source checks; no deployment writes."""
from pathlib import Path
import hashlib
import importlib
import json
import sys

HERE=Path(__file__).resolve().parent
PACKAGE="l45_corner_shared_20261025"

def verify():
    manifest=json.loads((HERE/"manifest.json").read_text())
    for relative,expected in manifest["files"].items():
        if hashlib.sha256((HERE/relative).read_bytes()).hexdigest()!=expected:
            raise RuntimeError("Frozen estimator source changed: "+relative)
    for name,module in list(sys.modules.items()):
        if name==PACKAGE or name.startswith(PACKAGE+"."):
            path=Path(module.__file__).resolve()
            if not path.is_relative_to(HERE/"runtime"):
                raise RuntimeError("A different frozen estimator with this namespace is loaded")
    return manifest

def load_estimator(boxes,seed):
    verify()
    runtime=str(HERE/"runtime")
    if runtime not in sys.path:sys.path.insert(0,runtime)
    cls=importlib.import_module(PACKAGE+".corner_estimator").Estimator
    return cls(boxes,method="D",seed=seed,particles=64)

def observation(step,position,visible_target,remaining):
    verify()
    import numpy as np
    runtime=str(HERE/"runtime")
    if runtime not in sys.path:sys.path.insert(0,runtime)
    cls=importlib.import_module(PACKAGE+".local_pursuit").Observation
    return cls(step,np.asarray(position,float),None if visible_target is None else np.asarray(visible_target,float),remaining)
'''
    (frozen/'loader.py').write_text(loader)
    descriptor=dict(version=result.get('version','public_wall_turn_regions_v1'),arm='D',
        role='accepted shared research estimator; not a deployment replacement',
        particle_budget=64,grid_spacing=.5,sigma=.35,persistence=.9,important_region_mass=.01,
        public_inputs=['known boxes','seed','step','own position','target measurement only when visible','remaining time'],
        output='(64 positions, 64 importance weights); downstream consumers must preserve weights',
        scope='20x20 direct-motion one-target problem; target axis limit1; dt1; radius6 deterministic occlusion; horizon80',
        no_hidden_state=True,no_network=True,no_search_comparison=True,deployment_changed=False,
        acceptance_result=str(OUT/'confirmation_results.json'),acceptance_sha256=sha(OUT/'confirmation_results.json'),
        provenance=lineage)
    (frozen/'descriptor.json').write_text(json.dumps(descriptor,ensure_ascii=False,indent=2)+'\n')
    files={str(p.relative_to(frozen)):sha(p) for p in list(runtime.glob('*.py'))+[frozen/'loader.py',frozen/'descriptor.json']}
    (frozen/'manifest.json').write_text(json.dumps(dict(files=files),ensure_ascii=False,indent=2)+'\n')
    (frozen/'README.md').write_text('''# 统一研究估计器：冻结D组

这是估计器验证通过后的研究快照，不修改部署、网络或既有实验。
通过`loader.py`的`load_estimator(boxes, seed)`创建，逐步调用`update(obs)`得到64个位置与对应重要性权重。
观测只含自身位置、时钟、剩余步数和可见时的目标测量；隐藏时目标必须为None。

静态图/历史初始化开销与正常更新均须计入未来预算。所有G/T/T+Geo必须共享此版本、相同历史及权重语义；不能把64个带权样本当作等权样本。
当前没有接入或运行搜索对照，也没有宣称捕获率改善、概率校准或50ms闭环保证。
''')
    print('Frozen accepted research estimator:',frozen)


if __name__=='__main__':main()
