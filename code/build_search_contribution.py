"""Private fixed-compatible solver: pursuer ablation, direct G and real deadlines."""
from pathlib import Path
import json
import shutil
import subprocess
import sys
from build_deployment_search import digest,replace_once

CODE=Path(__file__).resolve().parent
BUILD=CODE/'cpp/build_search_contribution'

def main():
    import pybind11
    parent=CODE/'cpp/build_candidate_ablation';manifest=json.loads((parent/'build_manifest.json').read_text())
    for name,sha in manifest['files'].items():
        if digest(parent/'source'/name)!=sha:raise ValueError('Frozen candidate source changed: '+name)
    source={n:(parent/'source'/n).read_text() for n in manifest['files']}
    for n in ('bindings.cpp','CMakeLists.txt'):source[n]=source[n].replace('bindings_candidate_ablation','bindings_search_contribution')
    code=source['solvers/interception_puct.hpp'].replace('InterceptionPUCT','ContributionPUCT')
    code=replace_once(code,'int last_source=0; InterceptionGuide guide; int candidate_mode=0;',
        'int last_source=0; InterceptionGuide guide; int candidate_mode=0;\n'
        '    bool no_pursuer_neural=false,direct_geometry=false; double limit_ms=0.;\n'
        '    long long team_policies[2]={0,0},discarded_noise_pairs=0;')
    code=replace_once(code,'int r=guide.pursuer;++policies;', 'int r=guide.pursuer;++policies;++team_policies[1];')
    code=replace_once(code,'last_source=guided?2:(neural?1:0);',
        'last_source=guided?2:((neural&&!(no_pursuer_neural&&guide.active&&team==1))?1:0);')
    code=replace_once(code,'''                ++policies;
                a.segment(p->m_action_idxs[r][0],p->m_action_idxs[r].size())=
                    m_policy_network_wrappers[r].policy_network->eval(p,p->policy_encoding(s,r),r,g_gen);''', '''                if(no_pursuer_neural&&guide.active&&team==1) {
                    std::normal_distribution<float> normal(0.f,1.f);
                    for(int axis=0;axis<(int)p->m_action_idxs[r].size();++axis)normal(g_gen);
                    ++discarded_noise_pairs;
                } else {
                    ++policies;++team_policies[team];
                    a.segment(p->m_action_idxs[r][0],p->m_action_idxs[r].size())=
                        m_policy_network_wrappers[r].policy_network->eval(p,p->policy_encoding(s,r),r,g_gen);
                }''')
    code=replace_once(code,'ContributionPUCT(bool proposal=false,bool rollout=false,int mode=0):proposal_enabled(proposal),rollout_enabled(rollout),candidate_mode(mode){}',
        'ContributionPUCT(bool proposal=false,bool rollout=false,int mode=0,bool no_neural=false,bool direct=false):proposal_enabled(proposal),rollout_enabled(rollout),candidate_mode(mode),no_pursuer_neural(no_neural),direct_geometry(direct){}\n'
        '    Vec sample_probe(Problem* p,const Vec& state,int team) {guide.prepare(dynamic_cast<Example8*>(p),state);return sample(p,state,team,false);}\n'
        '    void set_time_limit_ms(double value) {if(!std::isfinite(value)||value<0)throw std::invalid_argument("invalid time limit");limit_ms=value;}')
    code=replace_once(code,'policies=values=0;nodes.clear();nodes.reserve(settings.num_simulations+1);', '''policies=values=0;team_policies[0]=team_policies[1]=discarded_noise_pairs=0;
        if(direct_geometry) {
            if(!guide.active)return result;
            Eigen::Vector2f action=Eigen::Vector2f::Zero();
            bool valid=RootCandidateAblation::base(guide,state,3,Eigen::Vector2f::Zero(),action);
            if(!valid) {action.setZero();++guide.stats["direct_fallback"];}
            result.best_action=Vec::Zero(p->m_action_dim);
            result.best_action.segment(p->m_action_idxs[guide.pursuer][0],2)=action;
            result.success=true;result.num_visits=0;result.value=Vec::Zero(4);
            result.root_audit=Eigen::MatrixXf::Zero(1,p->m_action_dim+6);result.root_audit(0,5)=1;
            result.root_audit.row(0).tail(p->m_action_dim)=result.best_action.transpose();
            result.root_sources={2};guide.stats["selected_source"]=2;
            guide.stats["policy_pursuer"]=0;guide.stats["policy_evader"]=0;
            guide.stats["direct_geometry"]=1;guide.stats["search_ms"]=InterceptionGuide::milliseconds(search_start)-guide.stats["graph_build_ms"]-guide.stats["plan_ms"];
            result.guidance_stats=guide.stats;result.guidance_routes=guide.audit();result.guidance_paths=guide.paths();
            return result;
        }
        nodes.clear();nodes.reserve(settings.num_simulations+1);''')
    code=replace_once(code,'for (int simulation=0;simulation<settings.num_simulations;++simulation) {', '''int completed=0;
        for (int simulation=0;simulation<settings.num_simulations;++simulation) {
            if(completed>0&&limit_ms>0&&InterceptionGuide::milliseconds(search_start)>=limit_ms)break;''')
    code=replace_once(code,'            sum+=value;', '            sum+=value;++completed;')
    code=replace_once(code,'result.num_visits=settings.num_simulations;result.value=sum/settings.num_simulations;', '''guide.stats["policy_evader"]=team_policies[0];guide.stats["policy_pursuer"]=team_policies[1];
        guide.stats["discarded_neural_noise_pairs"]=discarded_noise_pairs;
        guide.stats["time_limit_ms"]=limit_ms;guide.stats["simulation_cap_hit"]=limit_ms>0&&completed==settings.num_simulations;
        result.guidance_stats=guide.stats;
        result.num_visits=completed;result.value=sum/completed;''')
    source['solvers/contribution_puct.hpp']=code
    source['solvers/solver_wrapper.hpp']=replace_once(source['solvers/solver_wrapper.hpp'],
        '#include "interception_puct.hpp"','#include "interception_puct.hpp"\n#include "contribution_puct.hpp"')
    source['solvers/solver_wrapper.hpp']=replace_once(source['solvers/solver_wrapper.hpp'],
        'if(string=="C_PUCT_ABLATION_F") {',
        'if(string=="C_CONTRIBUTION_S") {solver=new ContributionPUCT();}\n'
        '            else if(string=="C_CONTRIBUTION_F") {solver=new ContributionPUCT(true,false,1);}\n'
        '            else if(string=="C_CONTRIBUTION_G") {solver=new ContributionPUCT(true,false,3);}\n'
        '            else if(string=="C_CONTRIBUTION_N") {solver=new ContributionPUCT(true,false,3,true);}\n'
        '            else if(string=="C_CONTRIBUTION_GEOMETRY") {solver=new ContributionPUCT(true,false,3,false,true);}\n'
        '            else if(string=="C_PUCT_ABLATION_F") {')
    source['bindings.cpp']=replace_once(source['bindings.cpp'],'.def("set_seed",[](Solver_Wrapper &w,unsigned int seed) {w.solver->g_gen.seed(seed);})',
        '.def("set_time_limit_ms",[](Solver_Wrapper &w,double ms) {auto* s=dynamic_cast<ContributionPUCT*>(w.solver);if(!s)throw std::invalid_argument("not a contribution solver");s->set_time_limit_ms(ms);})\n'
        '        .def("set_seed",[](Solver_Wrapper &w,unsigned int seed) {w.solver->g_gen.seed(seed);})')
    source['bindings.cpp']=replace_once(source['bindings.cpp'],'    m.def("cpp_search",', '''    m.def("contribution_sample_probe",[](Problem_Wrapper& w,Solver_Wrapper& solver,Eigen::VectorXf state,int team) {
        auto* s=dynamic_cast<ContributionPUCT*>(solver.solver);
        if(!s)throw std::invalid_argument("invalid probe solver");
        return s->sample_probe(w.problem,state,team);
    });
    m.def("cpp_search",''')
    for n,s in source.items():
        path=BUILD/'source'/n;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(s)
    subprocess.run(['cmake','-S',str(BUILD/'source'),'-B',str(BUILD/'cmake'),'-DCMAKE_BUILD_TYPE=Release',
        f'-DPYTHON_EXECUTABLE={sys.executable}',f'-Dpybind11_DIR={pybind11.get_cmake_dir()}'],check=True)
    subprocess.run(['cmake','--build',str(BUILD/'cmake'),'-j','2'],check=True)
    binary=next((BUILD/'cmake').glob('bindings_search_contribution*.so'));shutil.copyfile(binary,BUILD/binary.name)
    (BUILD/'build_manifest.json').write_text(json.dumps(dict(parent=digest(parent/'build_manifest.json'),
        files={n:digest(BUILD/'source'/n) for n in source},binary=digest(BUILD/binary.name),builder=digest(Path(__file__))),indent=2)+'\n')

if __name__=='__main__':main()
