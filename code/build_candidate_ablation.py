"""Copy frozen interception search and add isolated F/D/G root proposals."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
from build_deployment_search import digest,replace_once

CODE=Path(__file__).resolve().parent
BUILD=CODE/'cpp/build_candidate_ablation'

def main():
    import pybind11
    parent=CODE/'cpp/build_interception'
    manifest=json.loads((parent/'build_manifest.json').read_text())
    for name,sha in manifest['files'].items():
        if digest(parent/'source'/name)!=sha:raise ValueError('Frozen source changed: '+name)
    sources={n:(parent/'source'/n).read_text() for n in manifest['files']}
    for name in ('bindings.cpp','CMakeLists.txt'):
        sources[name]=sources[name].replace('bindings_interception','bindings_candidate_ablation')
    sources['solvers/root_candidate_ablation.hpp']=(CODE/'cpp/solvers/root_candidate_ablation.hpp').read_text()
    code=sources['solvers/interception_puct.hpp']
    code=replace_once(code,'#include "interception_guidance.hpp"','#include "root_candidate_ablation.hpp"')
    code=replace_once(code,'int last_source=0; InterceptionGuide guide;',
        'int last_source=0; InterceptionGuide guide; int candidate_mode=0;')
    code=replace_once(code,'if(guided) {bool ok=false;intercept=guide.proposal(s,g_gen,ok);guided=ok;}', '''if(guided) {
            bool ok=false;
            if(candidate_mode==0)intercept=guide.proposal(s,g_gen,ok);
            else {
                Eigen::Vector2f mean=Eigen::Vector2f::Zero();
                if(candidate_mode==1) {
                    int r=guide.pursuer;++policies;
                    mean=RootCandidateAblation::policy_mean(m_policy_network_wrappers[r].policy_network,p,s,r);
                }
                intercept=RootCandidateAblation::proposal(guide,s,candidate_mode,mean,g_gen,ok);
            }
            guided=ok;
        }''')
    code=replace_once(code,'InterceptionPUCT(bool proposal=false,bool rollout=false):proposal_enabled(proposal),rollout_enabled(rollout){}',
        'InterceptionPUCT(bool proposal=false,bool rollout=false,int mode=0):proposal_enabled(proposal),rollout_enabled(rollout),candidate_mode(mode){}')
    sources['solvers/interception_puct.hpp']=code
    sources['solvers/solver_wrapper.hpp']=replace_once(sources['solvers/solver_wrapper.hpp'],
        'if(string=="C_PUCT_INTERCEPT_OFF") {',
        'if(string=="C_PUCT_ABLATION_F") {solver=new InterceptionPUCT(true,false,1);}\n'
        '            else if(string=="C_PUCT_ABLATION_D") {solver=new InterceptionPUCT(true,false,2);}\n'
        '            else if(string=="C_PUCT_ABLATION_G") {solver=new InterceptionPUCT(true,false,3);}\n'
        '            else if(string=="C_PUCT_INTERCEPT_OFF") {')
    sources['bindings.cpp']=replace_once(sources['bindings.cpp'],'    m.def("cpp_search",', '''    m.def("candidate_debug",[](Problem_Wrapper& w,Eigen::VectorXf state,int mode,Eigen::Vector2f mean) {
        InterceptionGuide guide;guide.prepare(dynamic_cast<Example8*>(w.problem),state);
        Eigen::Vector2f action=Eigen::Vector2f::Zero();
        bool valid=guide.active&&RootCandidateAblation::base(guide,state,mode,mean,action);
        pybind11::dict out;out["active"]=guide.active;out["action"]=action;out["valid"]=valid;return out;
    });
    m.def("candidate_mean_debug",[](Problem_Wrapper& w,Solver_Wrapper& solver,Eigen::VectorXf state,int robot) {
        return RootCandidateAblation::policy_mean(solver.solver->m_policy_network_wrappers[robot].policy_network,w.problem,state,robot);
    });
    m.def("cpp_search",''')
    for name,content in sources.items():
        path=BUILD/'source'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(content)
    subprocess.run(['cmake','-S',str(BUILD/'source'),'-B',str(BUILD/'cmake'),'-DCMAKE_BUILD_TYPE=Release',
        f'-DPYTHON_EXECUTABLE={sys.executable}',f'-Dpybind11_DIR={pybind11.get_cmake_dir()}'],check=True)
    subprocess.run(['cmake','--build',str(BUILD/'cmake'),'-j','2'],check=True)
    binary=next((BUILD/'cmake').glob('bindings_candidate_ablation*.so'));shutil.copyfile(binary,BUILD/binary.name)
    (BUILD/'build_manifest.json').write_text(json.dumps(dict(parent=digest(parent/'build_manifest.json'),
        files={n:digest(BUILD/'source'/n) for n in sources},binary=digest(BUILD/binary.name),
        builder=digest(Path(__file__))),indent=2)+'\n')

if __name__=='__main__':main()
