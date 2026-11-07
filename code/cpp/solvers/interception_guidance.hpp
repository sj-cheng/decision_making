#pragma once
#include "../problems/example8.hpp"
#include <array>
#include <chrono>
#include <map>
#include <numeric>

// Geometry is a proposal/rollout heuristic. The simulator remains unchanged.
class InterceptionGuide {
public:
    using V = Eigen::Vector2f;
    using Clock = std::chrono::steady_clock;
    struct Target { V point; int robot; std::vector<int> tail, end; };
    struct Path { int steps=1000000; std::vector<V> points; };
    struct Route {
        Target escape, intercept;
        int intercept_step=0;
        float margin=0;
        bool has_intercept=false;
        std::vector<std::array<float,12>> audit;
        std::vector<V> escape_path, pursuit_path;
    };
    Example8* p=nullptr;
    int evader=-1, pursuer=-1, remaining=0;
    float root_time=0;
    bool active=false;
    std::vector<Route> routes;
    std::map<std::string,double> stats;

private:
    const int inf=1000000;
    std::vector<float> signature;
    std::vector<V> vertices;
    std::array<std::vector<std::vector<int>>,2> distances, next;
    std::array<V,2> limits;
    int role(int robot) const { return robot<2?0:1; }
    int edge(const V& a,const V& b,int robot) const {
        V d=(b-a).cwiseAbs().cwiseQuotient(limits[role(robot)]);
        return std::max(0,(int)std::ceil(d.maxCoeff()-1e-6f));
    }
    V position(const Eigen::VectorXf& state,int robot) const {
        return state.segment(p->m_state_idxs[robot][0],2);
    }
    void build() {
        std::vector<float> key={p->m_dist,(float)evader,(float)pursuer};
        for(int i=0;i<2;++i) for(int j=0;j<2;++j) key.push_back(p->m_state_lims(i,j));
        for(int robot:{evader,pursuer}) for(int i=0;i<2;++i)
            key.push_back(p->m_action_lims(p->m_action_idxs[robot][i],1));
        for(const auto& box:p->m_obstacles) for(int i=0;i<2;++i) for(int j=0;j<2;++j) key.push_back(box(i,j));
        if(key==signature) return;
        auto began=Clock::now(); signature=key; vertices.clear();
        const float offset=.05f*p->m_dist;
        auto add=[&](V q) { if(legal(q)) vertices.push_back(q); };
        for(const auto& box:p->m_obstacles) for(int x=0;x<2;++x) for(int y=0;y<2;++y)
            add(V(box(0,x)+(x?offset:-offset),box(1,y)+(y?offset:-offset)));
        float l=p->m_state_lims(0,0)+offset,r=p->m_state_lims(0,1)-offset;
        float b=p->m_state_lims(1,0)+offset,t=p->m_state_lims(1,1)-offset;
        for(V q:{V(l,b),V(l,t),V(r,b),V(r,t),V((l+r)/2,b),V((l+r)/2,t),V(l,(b+t)/2),V(r,(b+t)/2)}) add(q);
        const int n=vertices.size();
        std::vector<std::vector<bool>> visible(n,std::vector<bool>(n));
        for(int i=0;i<n;++i) for(int j=0;j<n;++j) visible[i][j]=i==j||clear(vertices[i],vertices[j]);
        for(int team=0;team<2;++team) {
            int robot=team==0?evader:pursuer;
            distances[team].assign(n,std::vector<int>(n,inf));
            next[team].assign(n,std::vector<int>(n,-1));
            for(int i=0;i<n;++i) for(int j=0;j<n;++j) if(visible[i][j]) {
                distances[team][i][j]=edge(vertices[i],vertices[j],robot); next[team][i][j]=j;
            }
            for(int k=0;k<n;++k) for(int i=0;i<n;++i) for(int j=0;j<n;++j)
                if(distances[team][i][k]+distances[team][k][j]<distances[team][i][j]) {
                    distances[team][i][j]=distances[team][i][k]+distances[team][k][j];
                    next[team][i][j]=next[team][i][k];
                }
        }
        stats["graph_build_ms"]=milliseconds(began);
    }
public:
    static double milliseconds(Clock::time_point began) {
        return std::chrono::duration<double,std::milli>(Clock::now()-began).count();
    }
    bool legal(const V& q) const {
        for(int i=0;i<2;++i) if(q(i)<p->m_state_lims(i,0)||q(i)>p->m_state_lims(i,1)) return false;
        for(const auto& box:p->m_obstacles)
            if(q(0)>=box(0,0)&&q(0)<=box(0,1)&&q(1)>=box(1,0)&&q(1)<=box(1,1)) return false;
        return q.allFinite();
    }
    bool clear(const V& a,const V& b) {
        ++stats["geometry_queries"];
        if(!legal(a)||!legal(b)) return false;
        V d=b-a;
        for(const auto& box:p->m_obstacles) {
            float lo=0,hi=1; bool hit=true;
            for(int i=0;i<2;++i) {
                if(std::abs(d(i))<1e-9f) { if(a(i)<box(i,0)||a(i)>box(i,1)) hit=false; }
                else {float x=(box(i,0)-a(i))/d(i),y=(box(i,1)-a(i))/d(i);
                    lo=std::max(lo,std::min(x,y));hi=std::min(hi,std::max(x,y));}
            }
            if(hit&&lo<=hi) return false;
        }
        return true;
    }
    Target target(const V& q,int robot) {
        Target out{q,robot,{}, {}}; const int n=vertices.size(),team=role(robot);
        out.tail.assign(n,inf);out.end.assign(n,-1);
        std::vector<int> last(n,inf);
        for(int j=0;j<n;++j) if(clear(vertices[j],q)) last[j]=edge(vertices[j],q,robot);
        for(int i=0;i<n;++i) for(int j=0;j<n;++j) {
            int cost=distances[team][i][j]+last[j];
            if(cost<out.tail[i]) {out.tail[i]=cost;out.end[i]=j;}
        }
        return out;
    }
    Path path(const V& from,const Target& to,bool full=true) {
        Path out; int first=-1,team=role(to.robot),n=vertices.size();
        if(clear(from,to.point)) out.steps=edge(from,to.point,to.robot);
        for(int i=0;i<n;++i) if(to.tail[i]<inf&&clear(from,vertices[i])) {
            int cost=edge(from,vertices[i],to.robot)+to.tail[i];
            if(cost<out.steps) {out.steps=cost;first=i;}
        }
        if(out.steps>=inf) return out;
        if(first>=0) {
            int at=first,end=to.end[first]; out.points.push_back(vertices[at]);
            if(full) for(int guard=0;at!=end&&guard<n;++guard) {
                at=next[team][at][end]; if(at<0) {out.steps=inf;out.points.clear();return out;}
                out.points.push_back(vertices[at]);
            }
        }
        if(full||first<0) out.points.push_back(to.point);
        return out;
    }
    V move(const V& from,const V& waypoint,int robot) const {
        V delta=waypoint-from;
        float scale=std::max(1.f,delta.cwiseAbs().cwiseQuotient(limits[role(robot)]).maxCoeff());
        return delta/scale;
    }
    bool step_to(const V& from,const Target& to,V& action) {
        auto route=path(from,to,false);
        if(route.steps>=inf||route.points.empty()) return false;
        V waypoint=route.points.front();
        // A graph vertex can coincide with the current point; skip it.
        if((waypoint-from).norm()<1e-5f) {
            route=path(from,to,true);
            auto it=std::find_if(route.points.begin(),route.points.end(),[&](const V& v){return (v-from).norm()>1e-5f;});
            waypoint=it==route.points.end()?to.point:*it;
        }
        action=move(from,waypoint,to.robot);
        return legal(from+action);
    }
    void prepare(Example8* problem,const Eigen::VectorXf& state) {
        p=problem;active=false;routes.clear();stats.clear();evader=pursuer=-1;
        if(!p) return;
        int ne=0,np=0;
        for(int r=0;r<4;++r) if(p->is_active(state,r)) {
            if(r<2) {evader=r;++ne;} else {pursuer=r;++np;}
        }
        if(ne!=1||np!=1) return;
        active=true;stats["active"]=1;root_time=state(p->m_time_idx);
        remaining=std::max(0,(int)std::floor((p->m_tf-root_time)/p->m_timestep+1e-5f));
        for(int r:{evader,pursuer}) for(int axis=0;axis<2;++axis)
            limits[role(r)](axis)=p->m_action_lims(p->m_action_idxs[r][axis],1);
        auto began=Clock::now();build();
        V e=position(state,evader),pr=position(state,pursuer);
        struct Candidate {int id,score;Target goal;Path route;V direction;};
        std::vector<Candidate> candidates;
        for(int i=0;i<(int)vertices.size();++i) {
            auto te=target(vertices[i],evader),tp=target(vertices[i],pursuer);
            auto er=path(e,te),pp=path(pr,tp,false);
            if(er.steps<2||er.steps>remaining||er.points.empty()||pp.steps>=inf) continue;
            V direction=er.points.front()-e;
            if(direction.norm()<1e-5f) continue;
            candidates.push_back({i,pp.steps-er.steps,te,er,direction.normalized()});
        }
        std::stable_sort(candidates.begin(),candidates.end(),[](const Candidate& a,const Candidate& b){return a.score>b.score;});
        std::vector<V> headings;
        for(const auto& c:candidates) {
            bool different=true;for(const auto& h:headings) if(h.dot(c.direction)>.8660254f) different=false;
            if(!different) continue;
            Route route;route.escape=c.goal;route.intercept=target(e,pursuer);
            int best_late=inf;V at=e;size_t waypoint=0;
            for(int t=1;t<=std::min(24,std::min(remaining,c.route.steps));++t) {
                while(waypoint<c.route.points.size()&&(at-c.route.points[waypoint]).norm()<1e-5f) ++waypoint;
                if(waypoint==c.route.points.size()) break;
                at+=move(at,c.route.points[waypoint],evader);
                if(t!=1&&t!=2&&t!=4&&t!=8&&t!=12&&t!=16&&t!=24) continue;
                auto intercept=target(at,pursuer);int tp=path(pr,intercept,false).steps;
                route.audit.push_back({(float)routes.size(),(float)t,at(0),at(1),at(0),at(1),(float)tp,(float)(t-tp),0,c.goal.point(0),c.goal.point(1),0});
                if(tp>remaining) continue;
                int late=tp-t;
                if(!route.has_intercept || (best_late>0&&(late<=0||late<best_late))) {
                    route.intercept=intercept;route.intercept_step=t;route.margin=t-tp;
                    route.has_intercept=true;best_late=late;
                }
            }
            for(auto& row:route.audit) {row[8]=route.has_intercept&&row[1]==route.intercept_step;row[11]=route.has_intercept;}
            route.escape_path.push_back(e);
            route.escape_path.insert(route.escape_path.end(),c.route.points.begin(),c.route.points.end());
            route.pursuit_path.push_back(pr);
            auto pp=path(pr,route.intercept);
            route.pursuit_path.insert(route.pursuit_path.end(),pp.points.begin(),pp.points.end());
            routes.push_back(std::move(route));headings.push_back(c.direction);
            if(routes.size()==3) break;
        }
        stats["routes"]=routes.size();stats["plan_ms"]=milliseconds(began)-stats["graph_build_ms"];
    }
    Eigen::MatrixXf audit() const {
        size_t n=0;for(const auto& r:routes)n+=r.audit.size();
        Eigen::MatrixXf out(n,12);int i=0;
        for(const auto& r:routes)for(const auto& row:r.audit){for(int j=0;j<12;++j)out(i,j)=row[j];++i;}
        return out;
    }
    Eigen::MatrixXf paths() const {
        size_t n=0;for(const auto& r:routes)n+=r.escape_path.size()+r.pursuit_path.size();
        Eigen::MatrixXf out(n,4);int row=0;
        for(int i=0;i<(int)routes.size();++i) for(int kind=0;kind<2;++kind) {
            const auto& points=kind==0?routes[i].escape_path:routes[i].pursuit_path;
            for(const auto& q:points) {out(row,0)=i;out(row,1)=kind;out(row,2)=q(0);out(row,3)=q(1);++row;}
        }
        return out;
    }
    V proposal(const Eigen::VectorXf& state,std::default_random_engine& gen,bool& ok) {
        V at=position(state,pursuer),action=V::Zero();
        if(routes.empty()) {++stats["proposal_no_route"];ok=step_to(at,target(position(state,evader),pursuer),action);}
        else {int id=std::min((int)routes.size()-1,(int)(p->dist(gen)*routes.size()));
            ok=step_to(at,routes[id].intercept,action);if(!routes[id].has_intercept) ++stats["proposal_no_intercept"];}
        if(!ok){++stats["proposal_fallback"];return action;}
        V noisy=action;std::normal_distribution<float> normal(0,1);
        for(int i=0;i<2;++i) noisy(i)=std::max(-limits[1](i),std::min(limits[1](i),action(i)+.1f*limits[1](i)*normal(gen)));
        if(!legal(at+noisy)) {++stats["noise_fallback"];return action;}
        return noisy;
    }
    bool joint(const Eigen::VectorXf& state,int route_id,Eigen::VectorXf& action) {
        if(route_id<0||route_id>=(int)routes.size()||!p->is_active(state,evader)||!p->is_active(state,pursuer)) return false;
        const auto& route=routes[route_id];V e=position(state,evader),pr=position(state,pursuer),ae,ap;
        if((e-route.escape.point).norm()<1e-5f) {++stats["rollout_goal_reached"];return false;}
        if(!step_to(e,route.escape,ae)) return false;
        if(!route.has_intercept||state(p->m_time_idx)>=root_time+route.intercept_step*p->m_timestep)
            {if(!step_to(pr,target(e,pursuer),ap))return false;}
        else if(!step_to(pr,route.intercept,ap))return false;
        action=Eigen::VectorXf::Zero(p->m_action_dim);
        action.segment(p->m_action_idxs[evader][0],2)=ae;
        action.segment(p->m_action_idxs[pursuer][0],2)=ap;
        return true;
    }
};
