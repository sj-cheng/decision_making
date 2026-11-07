#pragma once
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <algorithm>

namespace local_geometry {
using Boxes=std::vector<Eigen::Matrix<double,2,2>>;
inline bool intersects(const Eigen::Vector2d& a,const Eigen::Vector2d& b,const Eigen::Matrix<double,2,2>& box) {
    double lo=0.,hi=1.;
    for(int axis=0;axis<2;++axis) {
        double d=b(axis)-a(axis);
        if(std::abs(d)<1e-12) {if(a(axis)<box(axis,0)||a(axis)>box(axis,1)) return false;}
        else {double x=(box(axis,0)-a(axis))/d,y=(box(axis,1)-a(axis))/d;
            lo=std::max(lo,std::min(x,y));hi=std::min(hi,std::max(x,y));}
    }
    return lo<=hi;
}
inline bool visible(const Eigen::Vector2d& point,const Eigen::Vector2d& origin,const Boxes& boxes,double range) {
    if((point-origin).norm()>range) return false;
    for(const auto& box:boxes) if(intersects(origin,point,box)) return false;
    return true;
}
inline Eigen::Array<bool,Eigen::Dynamic,1> legal(const Eigen::MatrixXd& points,const Boxes& boxes,
                                             const Eigen::Matrix<double,2,2>& bounds,double margin) {
    Eigen::Array<bool,Eigen::Dynamic,1> result(points.rows());
    for(int i=0;i<points.rows();++i) {
        bool good=true;
        for(int axis=0;axis<2;++axis) good &= points(i,axis)>bounds(axis,0)+margin && points(i,axis)<bounds(axis,1)-margin;
        for(const auto& box:boxes) good &= !(points(i,0)>=box(0,0)-margin && points(i,0)<=box(0,1)+margin &&
                                                           points(i,1)>=box(1,0)-margin && points(i,1)<=box(1,1)+margin);
        result(i)=good;
    }
    return result;
}
inline Eigen::Array<bool,Eigen::Dynamic,1> clear(const Eigen::MatrixXd& starts,const Eigen::MatrixXd& ends,const Boxes& boxes) {
    Eigen::Array<bool,Eigen::Dynamic,1> result(starts.rows());
    for(int i=0;i<starts.rows();++i) {
        bool good=true;
        for(const auto& box:boxes) if(intersects(starts.row(i),ends.row(i),box)) {good=false;break;}
        result(i)=good;
    }
    return result;
}
inline Eigen::Array<bool,Eigen::Dynamic,1> visibility(const Eigen::MatrixXd& points,const Eigen::MatrixXd& origins,
                                                   const Boxes& boxes,double range) {
    Eigen::Array<bool,Eigen::Dynamic,1> result(points.rows());
    for(int i=0;i<points.rows();++i) {
        result(i)=false;
        for(int j=0;j<origins.rows();++j) if(visible(points.row(i),origins.row(j),boxes,range)) {result(i)=true;break;}
    }
    return result;
}
inline Eigen::Array<bool,Eigen::Dynamic,1> negative(const Eigen::MatrixXd& points,const Eigen::MatrixXd& origins,
        const Eigen::VectorXd& active,const Eigen::MatrixXd& base,const Boxes& boxes,double range) {
    Eigen::Array<bool,Eigen::Dynamic,1> result(points.rows());
    for(int i=0;i<points.rows();++i) {
        result(i)=false;
        for(int j=0;j<origins.rows();++j) {
            if(active(j)<.5) continue;
            // Match float simulator coordinates, with double angular bins.
            Eigen::Vector2d delta=points.row(i).cast<float>().cast<double>().transpose()-origins.row(j).cast<float>().cast<double>().transpose();
            double distance=delta.norm();
            if(distance<=1e-8 || distance>range) continue;
            int beam=(int(std::nearbyint(std::atan2(delta(1),delta(0))*64/(2*3.14159265358979323846)))%64+64)%64;
            // Match the actual finite-beam sensor, not an ideal line of
            // sight. A beam can be occluded at a corner even when the line
            // to the candidate's center is clear.
            double obstacle=base(j,8+4*beam+2)*range;
            if(distance>=obstacle-1e-5) continue;
            double teammate=base(j,8+4*beam)*range;
            if(std::abs(distance-teammate)<=1e-5) continue;
            double observed=base(j,8+4*beam+1)*range;
            if(!(observed<range*(1.-1e-6) && distance>=observed-.65)) {result(i)=true;break;}
        }
    }
    return result;
}
}
