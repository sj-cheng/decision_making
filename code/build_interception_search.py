"""Build isolated interception search from the frozen deployment extension."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
from build_deployment_search import digest, replace_once

CODE=Path(__file__).resolve().parent
BUILD=CODE/'cpp/build_interception'

def main():
    import pybind11
    frozen=CODE/'cpp/build_deployment'
    manifest=json.loads((frozen/'build_manifest.json').read_text())
    for name,sha in manifest['generated'].items():
        if digest(frozen/'source'/name)!=sha:raise ValueError('Frozen source changed: '+name)
    source={n:(frozen/'source'/n).read_text() for n in manifest['generated']}
    source['bindings.cpp']=source['bindings.cpp'].replace('bindings_deployment','bindings_interception')
    source['CMakeLists.txt']=source['CMakeLists.txt'].replace('bindings_deployment','bindings_interception')
    source['solvers/interception_guidance.hpp']=(CODE/'cpp/solvers/interception_guidance.hpp').read_text()
    search=source['solvers/simultaneous_puct.hpp'].replace('SimultaneousPUCT','InterceptionPUCT')
    search=replace_once(search,'#include <map>','#include <map>\n#include "interception_guidance.hpp"')
    search=replace_once(search,'int visits=0; };','int visits=0; int source=0; };')
    search=replace_once(search,'long long policies=0, values=0;',
        'long long policies=0, values=0;\n    bool proposal_enabled=false, rollout_enabled=false;\n'
        '    int last_source=0; InterceptionGuide guide;')
    start,end=search.index('    Vec sample('),search.index('    int choose(')
    search=search[:start]+'''    Vec sample(Problem* p, const Vec& s, int team, bool root=false) {
        Vec a=p->sample_action(g_gen);
        const double draw=p->dist(g_gen);
        const bool neural=draw<settings.beta_policy;
        bool guided=proposal_enabled&&guide.active&&root&&team==1&&draw>=.5f&&draw<.75f;
        Eigen::Vector2f intercept=Eigen::Vector2f::Zero();
        if(guided) {bool ok=false;intercept=guide.proposal(s,g_gen,ok);guided=ok;}
        last_source=guided?2:(neural?1:0);
        for (int r=0;r<4;++r) {
            if (r/2!=team || !p->is_active(s,r)) {
                a.segment(p->m_action_idxs[r][0],p->m_action_idxs[r].size()).setZero();
            } else if(guided) {
                a.segment(p->m_action_idxs[r][0],2)=intercept;
            } else if (neural && m_policy_network_wrappers[r].valid) {
                ++policies;
                a.segment(p->m_action_idxs[r][0],p->m_action_idxs[r].size())=
                    m_policy_network_wrappers[r].policy_network->eval(p,p->policy_encoding(s,r),r,g_gen);
            }
        }
        return a;
    }
''' + search[end:]
    search=replace_once(search,'arm.action=sample(p,nodes[n].state,team);',
        'arm.action=sample(p,nodes[n].state,team,n==0); arm.source=last_source;\n'
        '            if(n==0&&team==1) {++guide.stats["root_candidates"];\n'
        '                ++guide.stats[std::string("root_source_")+std::to_string(last_source)];}')
    search=replace_once(search,'float discount=1.f;','''float discount=1.f;
            bool guided=false; int route=-1,prefix=0;
            if(rollout_enabled&&guide.active) {
                ++guide.stats["rollouts"];
                guided=p->dist(g_gen)<.8f;
                if(guided&&!guide.routes.empty()) {
                    route=std::min((int)guide.routes.size()-1,(int)(p->dist(g_gen)*guide.routes.size()));
                    ++guide.stats["guided_rollouts"];
                } else {guided=false;if(guide.routes.empty())++guide.stats["rollout_no_route"];}
            }''')
    search=replace_once(search,'Vec a=p->sample_rollout_action(s,g_gen);','''Vec a;
                bool used=false;
                if(guided&&d<8) {
                    used=guide.joint(s,route,a);
                    if(!used) {guided=false;++guide.stats["rollout_fallback"];}
                }
                if(!used) a=p->sample_rollout_action(s,g_gen);
                else {++prefix;++guide.stats["guided_steps"];
                    guide.stats["max_guided_prefix"]=std::max(guide.stats["max_guided_prefix"],(double)prefix);}''')
    search=replace_once(search,'public:\n    void set_params','''public:
    InterceptionPUCT(bool proposal=false,bool rollout=false):proposal_enabled(proposal),rollout_enabled(rollout){}
    void set_params''')
    search=replace_once(search,'policies=values=0;nodes.clear();',
        'auto search_start=InterceptionGuide::Clock::now();\n'
        '        if(proposal_enabled||rollout_enabled) guide.prepare(dynamic_cast<Example8*>(p),state);\n'
        '        else {guide.active=false;guide.routes.clear();guide.stats.clear();}\n'
        '        policies=values=0;nodes.clear();')
    search=replace_once(search,'result.root_audit=Eigen::MatrixXf::Zero(arms.size(),p->m_action_dim+6);',
        'result.root_audit=Eigen::MatrixXf::Zero(arms.size(),p->m_action_dim+6);\n'
        '        result.root_sources.resize(arms.size());')
    search=replace_once(search,'result.root_audit(i,0)=i;',
        'result.root_sources[i]=arms[i].source; result.root_audit(i,0)=i;')
    search=replace_once(search,'result.num_visits=settings.num_simulations;',
        'guide.stats["selected_source"]=arms[best].source;\n'
        '        guide.stats["search_ms"]=InterceptionGuide::milliseconds(search_start)-guide.stats["graph_build_ms"]-guide.stats["plan_ms"];\n'
        '        result.guidance_stats=guide.stats;result.guidance_routes=guide.audit();result.guidance_paths=guide.paths();\n'
        '        result.num_visits=settings.num_simulations;')
    source['solvers/interception_puct.hpp']=search
    source['solvers/solver.hpp']=replace_once(source['solvers/solver.hpp'],'Eigen::MatrixXf root_audit;',
        'Eigen::MatrixXf root_audit;\n        std::map<std::string,double> guidance_stats;\n'
        '        Eigen::MatrixXf guidance_routes, guidance_paths; std::vector<int> root_sources;')
    source['solvers/solver.hpp']='#include <map>\n#include <string>\n'+source['solvers/solver.hpp']
    source['solvers/solver_wrapper.hpp']=replace_once(source['solvers/solver_wrapper.hpp'],
        '#include "simultaneous_puct.hpp"','#include "simultaneous_puct.hpp"\n#include "interception_puct.hpp"')
    source['solvers/solver_wrapper.hpp']=replace_once(source['solvers/solver_wrapper.hpp'],
        'if (string == "C_PUCT_SYNC") {',
        'if(string=="C_PUCT_INTERCEPT_OFF") {solver=new InterceptionPUCT();}\n'
        '            else if(string=="C_PUCT_INTERCEPT_P") {solver=new InterceptionPUCT(true,false);}\n'
        '            else if(string=="C_PUCT_INTERCEPT_R") {solver=new InterceptionPUCT(false,true);}\n'
        '            else if(string=="C_PUCT_INTERCEPT_PR") {solver=new InterceptionPUCT(true,true);}\n'
        '            else if (string == "C_PUCT_SYNC") {')
    source['bindings.cpp']=replace_once(source['bindings.cpp'],
        '.def_readonly("root_audit", &Solver_Result::root_audit)',
        '.def_readonly("root_audit", &Solver_Result::root_audit)\n'
        '        .def_readonly("num_visits", &Solver_Result::num_visits)\n'
        '        .def_readonly("guidance_stats", &Solver_Result::guidance_stats)\n'
        '        .def_readonly("guidance_routes", &Solver_Result::guidance_routes)\n'
        '        .def_readonly("guidance_paths", &Solver_Result::guidance_paths)\n'
        '        .def_readonly("root_sources", &Solver_Result::root_sources)')
    source['bindings.cpp']=replace_once(source['bindings.cpp'],'    m.def("cpp_search",', '''    m.def("interception_debug",[](Problem_Wrapper& w,Eigen::VectorXf state,Eigen::Vector2f goal,int robot) {
        InterceptionGuide g;g.prepare(dynamic_cast<Example8*>(w.problem),state);
        pybind11::dict out;out["active"]=g.active;out["routes"]=g.audit();out["stats"]=g.stats;
        if(g.active) {
            Eigen::Vector2f start=state.segment(w.problem->m_state_idxs[robot][0],2);
            auto target=g.target(goal,robot);auto path=g.path(start,target);
            Eigen::MatrixXf points(path.points.size(),2);
            for(int i=0;i<(int)path.points.size();++i)points.row(i)=path.points[i].transpose();
            Eigen::Vector2f action=Eigen::Vector2f::Zero();bool valid=g.step_to(start,target,action);
            out["steps"]=path.steps;out["path"]=points;out["action"]=action;out["valid"]=valid;
        }
        return out;
    });
    m.def("cpp_search",''')
    for name,content in source.items():
        path=BUILD/'source'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(content)
    subprocess.run(['cmake','-S',str(BUILD/'source'),'-B',str(BUILD/'cmake'),'-DCMAKE_BUILD_TYPE=Release',
        f'-DPYTHON_EXECUTABLE={sys.executable}',f'-Dpybind11_DIR={pybind11.get_cmake_dir()}'],check=True)
    subprocess.run(['cmake','--build',str(BUILD/'cmake'),'-j','2'],check=True)
    binary=next((BUILD/'cmake').glob('bindings_interception*.so'));shutil.copyfile(binary,BUILD/binary.name)
    (BUILD/'build_manifest.json').write_text(json.dumps(dict(parent=digest(frozen/'build_manifest.json'),
        files={n:digest(BUILD/'source'/n) for n in source},binary=digest(BUILD/binary.name),
        builder=digest(Path(__file__))),indent=2)+'\n')

if __name__=='__main__':main()
