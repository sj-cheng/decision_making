"""Frozen candidate solver plus M: physical neural mean without F's rescaling."""
from pathlib import Path
import json
import shutil
import subprocess
import sys
from build_deployment_search import digest,replace_once

CODE=Path(__file__).resolve().parent
BUILD=CODE/'cpp/build_amplitude_match'

def main():
    import pybind11
    parent=CODE/'cpp/build_candidate_ablation'
    record=json.loads((parent/'build_manifest.json').read_text())
    for name,sha in record['files'].items():
        if digest(parent/'source'/name)!=sha:raise ValueError('Frozen candidate source changed: '+name)
    sources={n:(parent/'source'/n).read_text() for n in record['files']}
    for n in ('bindings.cpp','CMakeLists.txt'):
        sources[n]=sources[n].replace('bindings_candidate_ablation','bindings_amplitude_match')
    n='solvers/root_candidate_ablation.hpp'
    sources[n]=replace_once(sources[n],'if(mode==1) {','if(mode==4) action=mean;\n        else if(mode==1) {')
    n='solvers/interception_puct.hpp'
    sources[n]=replace_once(sources[n],'if(candidate_mode==1) {','if(candidate_mode==1||candidate_mode==4) {')
    n='solvers/solver_wrapper.hpp'
    sources[n]=replace_once(sources[n],'if(string=="C_PUCT_ABLATION_F") {',
        'if(string=="C_PUCT_AMPLITUDE_M") {solver=new InterceptionPUCT(true,false,4);}\n'
        '            else if(string=="C_PUCT_ABLATION_F") {')
    sources['bindings.cpp']=replace_once(sources['bindings.cpp'],'    m.def("cpp_search",', '''    m.def("matched_proposal_debug",[](Problem_Wrapper& w,Eigen::VectorXf state,int mode,Eigen::Vector2f mean,unsigned int seed) {
        InterceptionGuide guide;guide.prepare(dynamic_cast<Example8*>(w.problem),state);
        Eigen::Vector2f base=Eigen::Vector2f::Zero();
        bool valid=guide.active&&RootCandidateAblation::base(guide,state,mode,mean,base);
        std::default_random_engine gen(seed);bool ok=false;Eigen::Vector2f action=Eigen::Vector2f::Zero();
        if(guide.active)action=RootCandidateAblation::proposal(guide,state,mode,mean,gen,ok);
        std::ostringstream rng;rng<<gen;pybind11::dict out;
        out["base"]=base;out["action"]=action;out["valid"]=valid;out["ok"]=ok;out["rng"]=rng.str();return out;
    });
    m.def("cpp_search",''')
    # These are the complete scientific changes; all perturbation/RNG code is copied.
    for name,content in sources.items():
        path=BUILD/'source'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(content)
    subprocess.run(['cmake','-S',str(BUILD/'source'),'-B',str(BUILD/'cmake'),'-DCMAKE_BUILD_TYPE=Release',
        f'-DPYTHON_EXECUTABLE={sys.executable}',f'-Dpybind11_DIR={pybind11.get_cmake_dir()}'],check=True)
    subprocess.run(['cmake','--build',str(BUILD/'cmake'),'-j','2'],check=True)
    binary=next((BUILD/'cmake').glob('bindings_amplitude_match*.so'));shutil.copyfile(binary,BUILD/binary.name)
    (BUILD/'build_manifest.json').write_text(json.dumps(dict(parent=digest(parent/'build_manifest.json'),
        files={n:digest(BUILD/'source'/n) for n in sources},binary=digest(BUILD/binary.name),builder=digest(Path(__file__))),indent=2)+'\n')

if __name__=='__main__':main()
