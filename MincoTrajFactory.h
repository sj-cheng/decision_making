#ifndef __MINCO_TRAJ_FACTORY_H__
#define __MINCO_TRAJ_FACTORY_H__

#pragma once

#include <Eigen/Dense>
#include <vector>

namespace RSG_SIM
{

constexpr int POLY_DIM = 2;                             // 维度
constexpr int POLY_CTRL_EFFORT = 3;                     // Control Effort 次数，2->Accelerate, 3->Jerk
constexpr int POLY_DEGREE = POLY_CTRL_EFFORT * 2 - 1;   // 轨迹多项式次数

typedef Eigen::Matrix<double, POLY_DEGREE+1, POLY_DIM>       CoffMat6x2d;
typedef Eigen::Matrix<double, POLY_CTRL_EFFORT*2, POLY_DIM>  QMat6x2d;
typedef Eigen::Vector<double, POLY_DEGREE+1>                 CoffVec6d;
typedef Eigen::Vector<float, POLY_DEGREE+1>                 CoffVec6f;
typedef Eigen::Vector<double, 2>                             TimeVec2d;
typedef Eigen::Matrix<double, POLY_DEGREE+1, POLY_DEGREE+1>  MMat6x6d;
typedef Eigen::Matrix<float, POLY_DEGREE+1, POLY_DEGREE+1>  MMat6x6f;
typedef Eigen::Matrix<double, POLY_DEGREE+1, POLY_DEGREE+1, Eigen::RowMajor>  MMat6x6dRowMajor;
typedef Eigen::Matrix<float, POLY_DEGREE+1, POLY_DEGREE+1, Eigen::RowMajor>  MMat6x6fRowMajor;

typedef Eigen::Vector<double, POLY_DIM>                      StateVec2;
typedef Eigen::Matrix<double, POLY_CTRL_EFFORT, POLY_DIM>    StateMat3x2;

class PolyTraj {
public:

    PolyTraj(CoffMat6x2d coff_ = CoffMat6x2d::Zero()) : coff(coff_) { }

    static CoffVec6d construct_beta(double t, int rank);

    static MMat6x6d construct_M(TimeVec2d ts);

    static PolyTraj init_by_qT(QMat6x2d q0, QMat6x2d qT, TimeVec2d ts);

    StateVec2 get_pos(double t) {
        CoffVec6d beta = construct_beta(t, 0);
        return beta.transpose() * coff;
    }

    StateVec2 get_vel(double t) {
        CoffVec6d beta = construct_beta(t, 1);
        return beta.transpose() * coff;
    }

    StateVec2 get_acc(double t) {
        CoffVec6d beta = construct_beta(t, 2);
        return beta.transpose() * coff;
    }

    CoffMat6x2d get_poly_coff() {
        return coff;
    }

private:
    CoffMat6x2d coff;
};

class MincoTrajFactory {
public:
    MincoTrajFactory() : pJpC_(MMat6x6d::Zero()) {}

    void clearPJpC() {
        pJpC_ = MMat6x6d::Zero();
    }

    MMat6x6d getMatrixMInv(TimeVec2d ts, int dof);

    PolyTraj solveWithCostJ(StateMat3x2 q0, StateMat3x2 qT, TimeVec2d ts, int dof);

    std::pair<StateVec2, StateVec2> solveEndPosEqu(
        TimeVec2d ts,
        int dof,
        double t,
        int rank,
        StateMat3x2 q0,
        StateVec2 velT,
        double max_min_equ_vel);

    std::pair<StateVec2, StateVec2> solveEndPosBound(TimeVec2d ts, StateMat3x2 q0, StateVec2 velT, int rank, double maxVel);

    /***
     * @brief 计算得到∫ββ^T
     */
    static MMat6x6d betabetaT_int(double t, int rank);
    static MMat6x6d betabetaT(double t, int rank);

    void add_control_effort_cost(double T);
    void add_acc_cost(double T);

private:
    MMat6x6d pJpC_;
    // MMat6x6d Minv_;
};

} // namespace RSG_SIM

#endif