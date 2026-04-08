#include <cmath>
#include <iostream>
#include "MincoTrajFactory.h"
#include "console_style.h"

namespace RSG_SIM
{

CoffVec6d PolyTraj::construct_beta(double t, int rank) {
    CoffVec6d betaT = CoffVec6d::Zero();
    CoffVec6d beta_coff = CoffVec6d::Zero();

    // beta:
    // [1 T T^2 T^3 ...] (rank=0)
    // [0 1 2*T 3*T^2 4*T^3 ...] (rank=1)
    // [0 0 2   2*3*T 3*4*T^2 ...] (rank=2)

    // 计算T有关的部分 [1 T T^2 T^3 ...]
    betaT(rank) = 1;
    for (int i = rank+1; i <= POLY_DEGREE; i++) {
        betaT(i) = betaT(i-1) * t;
    }
    // 计算多项式系数 [1 1 1 1 1 1](rank=0)     [0 1 2 3 4 ...](rank=1)     [0 0 2 2*3 3*4 ...](rank=2)
    for (int i = rank; i <= POLY_DEGREE; i++) {
        double coff = 1;
        for (int j = 0; j < rank; j++) {
            coff *= i-j;
        }
        beta_coff(i) = coff;
    }

    return betaT.cwiseProduct(beta_coff);
}

MMat6x6d PolyTraj::construct_M(TimeVec2d ts) {
    MMat6x6d M;
    int row_idx = 0;
    for (int i = 0; i < ts.size(); i++) {
        for (int s = 0; s < POLY_CTRL_EFFORT; s++) {
            M.row(row_idx) = construct_beta(ts(i), s).transpose();
            row_idx ++;
        }
    }
    return M;
}

PolyTraj PolyTraj::init_by_qT(QMat6x2d q0, QMat6x2d qT, TimeVec2d ts) {

    MMat6x6d M = construct_M(ts);
    MMat6x6d M_inv = M.inverse();

    QMat6x2d q;
    q << q0, qT;
    CoffMat6x2d c = M_inv * q;
    return PolyTraj(c);
}

MMat6x6d MincoTrajFactory::getMatrixMInv(TimeVec2d ts, int dof) {
    // 如果Minv已经有缓存，就用缓存过的
    // if (Minv_.size() != 0) {
    //     return Minv_;
    // }

    MMat6x6d M = PolyTraj::construct_M(ts);
    MMat6x6d M_inv = M.inverse();

    // ~q是约束中的自由变量，如当不限制速度和加速度时 ~q=[0,0,0,0,vT,aT]
    // 对末端a（即[0,0,0,0,0,a]^T向量）求偏导:∂J/∂(~q) = ∂q^T/∂(~q) * ∂c^T/∂q * ∂J/∂c
    // pQptQ = ∂q^T/∂(~q) = diag([0,0,0, ... ,1])
    // pCpQ = ∂c^T/∂(q) = M^(-T)
    // pJpC = ∂J/∂c = ∫(ββ^T)dt * c + ...
    // pQptQ 是q对tilt q的偏导数

    MMat6x6d pQptQ = MMat6x6d::Zero();
    pQptQ.bottomRightCorner(dof, dof) = Eigen::MatrixXd::Identity(dof, dof);

    MMat6x6d pCpQ = M_inv.transpose();
    MMat6x6d pJptQ = pQptQ * pCpQ * pJpC_;

    MMat6x6d newM = M;
    newM.bottomRows(dof) = pJptQ.bottomRows(dof);
    // std::cout << FG_YELLOW "pQptQ:\n" FG_DEFAULT << pQptQ << std::endl << std::endl;
    // std::cout << FG_YELLOW "pCpQ:\n" FG_DEFAULT << pCpQ << std::endl << std::endl;
    // std::cout << FG_YELLOW "pJpC_:\n" FG_DEFAULT << pJpC_ << std::endl << std::endl;

    MMat6x6d newM_inv = newM.inverse();

    return newM_inv;
}

PolyTraj MincoTrajFactory::solveWithCostJ(StateMat3x2 q0, StateMat3x2 qT, TimeVec2d ts, int dof) {
    // if (q0.rows() != Poly_Traj::Poly_Ctrl_Effort || qT.rows() != Poly_Traj::Poly_Ctrl_Effort-dof) {
    //     throw std::invalid_argument("q0 and qT should have S rows");
    // }
    // if (q0.cols() != Poly_Traj::Poly_Dim || qT.cols() != Poly_Traj::Poly_Dim) {
    //     throw std::invalid_argument("q0 and qT should have N columns");
    // }

    // qT.conservativeResize(Poly_Traj::Poly_Ctrl_Effort, Poly_Traj::Poly_Dim);
    QMat6x2d q;
    q << q0, qT;
    CoffMat6x2d c = getMatrixMInv(ts, dof) * q;

    // std::cout << "mat M=\n" << getMatrixMInv(ts, dof) << std::endl << "mat q=\n" << q << std::endl;
    return PolyTraj(c);
}

std::pair<StateVec2, StateVec2> MincoTrajFactory::solveEndPosEqu(
    TimeVec2d ts,
    int dof,
    double t,
    int rank,
    StateMat3x2 q0,
    StateVec2 velT,
    double max_min_equ_vel)
{
    // 求解不等式 (Phi'(t)^T) * (M^-1) * q （<= v_max 或 a_max，取第二行或第三行）或（>= -v_max 或 -a_max）
    // c0*p0+c1*v0+c2*a0+c3*pT+c4*vT <= v_max    (Minco求解时，末端aT是线性相关量)
    // aT 是线性相关（可被其他表出）
    // (v_max - c0*p0+c1*v0+c2*a0+c4*vT) / c3

    // 计算(Phi'(t)^T) * (M^-1)
    CoffVec6d phi_rank = PolyTraj::construct_beta(t, rank);
    MMat6x6d Minv = getMatrixMInv(ts, dof);
    CoffVec6d bound_coff_A = phi_rank.transpose() * Minv;
    // std::cout << "time t="<< t << " ,bound_coff_A=\n" << bound_coff_A << std::endl;

    // 构造系数向量[p0, v0, a0, 0, vT, 0]
    CoffMat6x2d q;
    q << q0, StateMat3x2::Zero();

    // 末端速度受约束
    q.block(POLY_CTRL_EFFORT+1, 0, 1, POLY_DIM) = velT.transpose();

    // [c0*p0, c1*v0, c2*a0, c3*pT, c4*vT, c5*0]，取出pT对应的系数
    double posT_coff_in_equ = bound_coff_A(POLY_CTRL_EFFORT);

    // 计算等式常数项(v_max - c0*p0+c1*v0+c2*a0+c4*vT)

    StateVec2 const_val_max_vec = StateVec2::Constant(max_min_equ_vel);

    // std::cout << "const_val_max_vec=\n"<< const_val_max_vec << " ,q=\n" << q << std::endl;
    StateVec2 const_val_max = -(bound_coff_A.transpose() * q).transpose() + const_val_max_vec;
    StateVec2 const_val_min = -(bound_coff_A.transpose() * q).transpose() - const_val_max_vec;

    // 考虑变号
    StateVec2 posUpper, posLower;
    if(posT_coff_in_equ > 0)
    {
        posUpper = const_val_max / posT_coff_in_equ;   // 上界
        posLower = const_val_min / posT_coff_in_equ;   // 下界
    }
    else
    {
        posUpper = const_val_min / posT_coff_in_equ;   // 上界
        posLower = const_val_max / posT_coff_in_equ;   // 下界
    }
    return std::make_pair(posUpper, posLower);
}


std::pair<StateVec2, StateVec2> MincoTrajFactory::solveEndPosBound(
    TimeVec2d ts,
    StateMat3x2 q0,
    StateVec2 velT,
    int rank,
    double maxVel)
{
    assert(rank == 1 || rank == 2);
    // 区间划分的段数
    int seg_count = 5;
    double den = ts[1] / (2*seg_count);

    Eigen::MatrixXd bound_u_results = Eigen::MatrixXd::Zero(seg_count, POLY_DIM);
    Eigen::MatrixXd bound_d_results = Eigen::MatrixXd::Zero(seg_count, POLY_DIM);

    for (int i = 0; i < seg_count; i++) {
        double t = (2*i+1) * den;
        auto [pos_u, pos_d] = solveEndPosEqu(ts, 1, t, rank, q0, velT, maxVel);
        bound_u_results.row(i) = pos_u;
        bound_d_results.row(i) = pos_d;
    }

    StateVec2 min_bound_u = bound_u_results.colwise().minCoeff();
    StateVec2 max_bound_d = bound_d_results.colwise().maxCoeff();

    return std::make_pair(min_bound_u, max_bound_d);
}

MMat6x6d MincoTrajFactory::betabetaT_int(double t, int rank)
{
    CoffVec6d b = PolyTraj::construct_beta(t, rank);
    MMat6x6d bbT = b * b.transpose();
    MMat6x6d int_coff = MMat6x6d::Zero();
    for (int i = 0; i <= POLY_DEGREE; i++)
    {
        for (int j = 0; j <= POLY_DEGREE; j++)
        {
            if (i+j-2*rank+1 > 0)
            {
                int_coff(i, j) = 1.0 / (i+j-2*rank+1);
            }
        }
    }
    return t * bbT.cwiseProduct(int_coff);
}

MMat6x6d MincoTrajFactory::betabetaT(double t, int rank) {
    CoffVec6d b = PolyTraj::construct_beta(t, rank);
    return b * b.transpose();
}

void MincoTrajFactory::add_control_effort_cost(double T) {
    // pJpC = ∂J/∂c = ∫(ββ^T)dt * c
    // 构造末端时间的代价矩阵∫(ββ^T)dt，维度(M+1)*(M+1)
    MMat6x6d bbT_int = betabetaT_int(T, POLY_CTRL_EFFORT);
    pJpC_ += bbT_int;
}

void MincoTrajFactory::add_acc_cost(double T) {
    // pJpC = ∂J/∂c = ∫(ββ^T)dt * c
    // 构造末端时间的代价矩阵∫(ββ^T)dt，维度(M+1)*(M+1)
    MMat6x6d bbT_int = betabetaT_int(T, POLY_CTRL_EFFORT-1);
    pJpC_ += bbT_int;
}

} // namespace RSG_SIM
