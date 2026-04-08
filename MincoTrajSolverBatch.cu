#include "MincoTrajFactory.h"
#include <glm/geometric.hpp>
#include <iostream>
#include "MincoTrajSolverBatch.cuh"
#include "cuda_helper.h"
#include "Time.h"
#include "common.cuh"

namespace RSG_SIM
{

__constant__ __device__ MincoTrajSolverParam d_mtsParams;
__constant__ __device__ MincoTrajMatrics d_mtsMatrics;


void setMincoTrajParams(const MincoTrajSolverParam& host_params)
{
    checkCudaErrors(cudaMemcpyToSymbol(d_mtsParams, &host_params, sizeof(d_mtsParams)));
}


void setMtsMatrics(const MincoTrajMatrics& val)
{
    checkCudaErrors(cudaMemcpyToSymbol(d_mtsMatrics, &val, sizeof(d_mtsMatrics)));
}


template<int I>
__device__ __host__ __forceinline__ void unroll(VBuffer::MTCBufferView &coff_cont, int rob_idx, glm::vec2 p0, glm::vec2 v0, glm::vec2 a0, glm::vec2 pT) {
    coff_cont.get<I>(rob_idx) =
        d_mtsMatrics.Minv[I][0] * p0 +
        d_mtsMatrics.Minv[I][1] * v0 +
        d_mtsMatrics.Minv[I][2] * a0 +
        d_mtsMatrics.Minv[I][3] * pT;
}

/**
 * @brief 循环展开矩阵乘法，非递归结束 I!=N 的情况
 */
template<int I, int N>
struct UnrollLoop {
    static __device__ __host__ __forceinline__ void call(VBuffer::MTCBufferView &coff_cont, int rob_idx, glm::vec2 p0, glm::vec2 v0, glm::vec2 a0, glm::vec2 pT) {
        unroll<I>(coff_cont, rob_idx, p0, v0, a0, pT);
        UnrollLoop<I + 1, N>::call(coff_cont, rob_idx, p0, v0, a0, pT);
    }
};


/**
 * @brief 处理递归结束 N,N=5,5 的情况
 */
template<int N>
struct UnrollLoop<N, N> {
    static __device__ __host__ __forceinline__ void call(VBuffer::MTCBufferView &coff_cont, int rob_idx, glm::vec2 p0, glm::vec2 v0, glm::vec2 a0, glm::vec2 pT) {
        unroll<N>(coff_cont, rob_idx, p0, v0, a0, pT);
    }
};


/**
 * @brief 计算1维情况下的MINCO轨迹，通过 [p0 v0 a0 pT vT 0] 计算Minco轨迹
 * 给定初值和终值，计算多项式轨迹系数
*/
__global__ void mincoTrajSolveKernel(
        const glm::vec2 *__restrict__ pos,
        const glm::vec2 *__restrict__ vel,
        const glm::vec2 *__restrict__ acc,
        const glm::vec2 *__restrict__ posT,
        VBuffer::MTCBufferView coff_cont)
{
    GET_CUDA_ID(idx, VConfig::AgentNum);

    // 将值复制到寄存器
    glm::vec2 p0 = pos[idx];
    glm::vec2 v0 = vel[idx];
    glm::vec2 a0 = acc[idx];
    glm::vec2 pT = posT[idx];
    // glm::vec2 vT = d_mtsParams.velDecayCoff * v0;

    UnrollLoop<0,5>().call(coff_cont, idx, p0, v0, a0, pT);

    // if(idx==0) {
    //     printf("Minv=\n");
    //     for(int i=0; i<6; i++) {
    //         for(int j=0; j<6; j++) {
    //             printf("%.2f\t\t", d_mtsMatrics.Minv[i][j]);
    //         }
    //         printf("\n");
    //     }
    //     printf("------------------------\n");
    // }

    // printf( "P0=" FG_YELLOW "(%f, %f)" FG_DEFAULT ", V0=" FG_BLUE "(%f, %f)" FG_DEFAULT ", A0=" FG_GREEN "(%f, %f)" FG_DEFAULT
    //         "PT=" FG_YELLOW "(%f, %f)" FG_DEFAULT LINE_ENDL,
    //         p0.x, p0.y, v0.x, v0.y, a0.x, a0.y, pT.x, pT.y);
    // printf("coff.x=[%f, %f, %f, %f, %f, %f]" LINE_ENDL "coff.y=[%f, %f, %f, %f, %f, %f]" LINE_ENDL,
    //         coff_cont.get<0>(idx).x, coff_cont.get<1>(idx).x, coff_cont.get<2>(idx).x,
    //         coff_cont.get<3>(idx).x, coff_cont.get<4>(idx).x, coff_cont.get<5>(idx).x,
    //         coff_cont.get<0>(idx).y, coff_cont.get<1>(idx).y, coff_cont.get<2>(idx).y,
    //         coff_cont.get<3>(idx).y, coff_cont.get<4>(idx).y, coff_cont.get<5>(idx).y);
}

/**
 *
 */
__device__ inline void solvePosBoundKernel(
        glm::vec2 p0,
        glm::vec2 v0,
        glm::vec2 a0,
        const float *__restrict__ bound_coff,
        float max_val,
        glm::vec2 &upper,
        glm::vec2 &lower)
{
    // 当前版本仅适用于vT无约束的情况
    assert(d_mtsParams.mincoDof == 2);

    float pT_coff = bound_coff[3];

    glm::vec2 lft_const_coff =
        p0*bound_coff[0] +
        v0*bound_coff[1] +
        a0*bound_coff[2];

    glm::vec2 pT_u, pT_d;

    // !分母小于0，则不等式方向改变，需要交换上下界。
    // pT_u与pT_d相差正好是max_val的符号
    if(pT_coff < 0)
        max_val = -max_val;

    pT_u = ( max_val - lft_const_coff) / pT_coff;
    pT_d = (-max_val - lft_const_coff) / pT_coff;

    // if(pT_coff < 0)
    //     swap(pT_u, pT_d);

    // 与当前界取交集
    upper.x = min(upper.x, pT_u.x);
    upper.y = min(upper.y, pT_u.y);
    lower.x = max(lower.x, pT_d.x);
    lower.y = max(lower.y, pT_d.y);
}

__global__ void applyVelAccConstrainKernel(
        const glm::vec2 *__restrict__ pos,
        const glm::vec2 *__restrict__ vel,
        const glm::vec2 *__restrict__ acc,
        glm::vec2       *__restrict__ upper,
        glm::vec2       *__restrict__ lower)
{
    GET_CUDA_ID(idx, VConfig::AgentNum);

    glm::vec2 p0 = pos[idx];
    glm::vec2 v0 = vel[idx];
    glm::vec2 a0 = acc[idx];
    // glm::vec2 vT = d_mtsParams.velDecayCoff * v0;

    glm::vec2 pos_upper {1e3, 1e3}, pos_lower {-1e3, -1e3};

    #pragma unroll
    for(int i=0; i<d_mtsParams.ckptCount; i++)
    {
        // TODO: 末端的有解
        solvePosBoundKernel(p0, v0, a0, d_mtsMatrics.velBoundCoff[i], d_mtsParams.maxVel, pos_upper, pos_lower);
        // !由于使用了目标速度控制，ACC控制器已经是受控的，所有没有使用ACC
        // solvePosBoundKernel(p0, v0, a0, d_mtsMatrics.accBoundCoff[i], d_mtsParams.maxAcc, pos_upper, pos_lower);
    }

    upper[idx] = pos_upper;
    lower[idx] = pos_lower;

    // if(idx == 0) {
    //     printf(FG_GREEN "Pos Upper:%.3f, %.3f" FG_DEFAULT LINE_ENDL, pos_upper.x, pos_upper.y);
    //     printf(FG_GREEN "Pos Lower:%.3f, %.3f" FG_DEFAULT LINE_ENDL, pos_lower.x, pos_lower.y);
    // }
}


/***
 * @brief 更新运动学参数，从轨迹计算目标速度
 * @param coff_cont 轨迹系数容器
 * @param h 时间步长
 * @param tgt_vel 目标速度
 */
__global__ void updateKinParamKernel(
        VBuffer::MTCBufferView coff_cont,
        double t,
        glm::vec2 *__restrict__ tgt_vel)
{
    GET_CUDA_ID(idx, VConfig::AgentNum);

    double t2 = t*t;
    double t3 = t2*t;
    double t4 = t3*t;
    //pos: a0 + a1*t + a2*t^2 + a3*t^3   + a4*t^4    + a5*t^5
    //vel: 0  + a1   + 2*a2*t + 3*a3*t^2 + 4*a4*t^3  + 5*a5*t^4
    //acc: 0  + 0    + 2*a2   + 6*a3*t   + 12*a4*t^2 + 20*a5*t^3

    // 目标速度控制
    // TODO. 优化 + 增强数值稳定
    tgt_vel[idx] =  coff_cont.get<1>(idx) +
                    2.0 * coff_cont.get<2>(idx) * t+
                    3.0 * coff_cont.get<3>(idx) * t2 +
                    4.0 * coff_cont.get<4>(idx) * t3 +
                    5.0 * coff_cont.get<5>(idx) * t4;

    // acc[idx] = 2.0f * coff_cont.get<2>(idx) +
    //                6.0f * coff_cont.get<3>(idx) * substep_time +
    //                12.0f * coff_cont.get<4>(idx) * sst_pow2 +
    //                20.0f * coff_cont.get<5>(idx) * sst_pow3;

    // if (glm::length(acc[idx]) >= 10)
    // {
    //     printf(BG_GREEN "Acc(%f) >= 10" FG_DEFAULT LINE_ENDL, acc[idx]);
    // }
    // printf(BG_MAGENTA "Require Traj @ time=%f!" FG_DEFAULT LINE_ENDL, t);
}

/*****************************************************************************************************/

void MincoTrajSolverBatch::mincoTrajSolve(
        const glm::vec2 *__restrict__ pos,
        const glm::vec2 *__restrict__ vel,
        const glm::vec2 *__restrict__ acc,
        const glm::vec2 *__restrict__ posT,
        const VBuffer::MTCBufferView &coff_container
)
{
    CUDA_KERNEL_PROFILE(
        VConfig::AgentNum,
        256,
        mincoTrajSolveKernel,
        pos, vel, acc, posT, coff_container
    );
}

void MincoTrajSolverBatch::applyVelAccConstrain(
        const glm::vec2 *__restrict__ pos,
        const glm::vec2 *__restrict__ vel,
        const glm::vec2 *__restrict__ acc,
        glm::vec2       *__restrict__ upper,
        glm::vec2       *__restrict__ lower)
{
    CUDA_KERNEL_PROFILE(
        VConfig::AgentNum,
        256,
        applyVelAccConstrainKernel,
        pos, vel, acc, upper, lower
    );
}

void MincoTrajSolverBatch::updateKinParam(
        VBuffer::MTCBufferView coff_cont,
        double t,
        glm::vec2 *__restrict__ tgt_vel)
{
    CUDA_KERNEL_PROFILE(
        VConfig::AgentNum,
        256,
        updateKinParamKernel,
        coff_cont, t, tgt_vel
    );
}

/*****************************************************************************************************/

void MincoTrajSolverBatch::setMincoTrajParams(const MincoTrajSolverParam& host_params)
{
    RSG_SIM::setMincoTrajParams(host_params);
    m_h_params = host_params;
}


void MincoTrajSolverBatch::reloadMincoRuntimeData()
{
    TimeVec2d ts {0, m_h_params.trajDur};

    MincoTrajFactory factory;
    // 求解得到Minv
    factory.add_control_effort_cost(m_h_params.trajDur);
    MMat6x6d eigen_mat_MInv = factory.getMatrixMInv(ts, MincoTrajSolverParam::mincoDof);

    // Eigen是列主导
    Eigen::Map<MMat6x6fRowMajor> map_mat(&m_h_matrics.Minv[0][0]);

    // memcpy(m_h_matrics.Minv, eigen_mat_MInv.transpose().data(), sizeof(m_h_matrics.Minv));
    map_mat = eigen_mat_MInv.cast<float>();
    // std::cout << FG_YELLOW "Minv:\n" FG_DEFAULT << eigen_mat_MInv << std::endl << std::endl;

    // 在5个检查点上，生成对应的 (β^T*M^-1)
    for(int i=0; i<m_h_params.ckptCount; i++)
    {
        double t = (2*i+1)*(MincoTrajSolverParam::trajDur / (m_h_params.ckptCount*2));
        CoffVec6d vel_phi_rank = RSG_SIM::PolyTraj::construct_beta(t, 1);     // 速度检查
        CoffVec6d acc_phi_rank = RSG_SIM::PolyTraj::construct_beta(t, 2);     // 加速度检查

        CoffVec6d vel_bound_coff_A = vel_phi_rank.transpose() * eigen_mat_MInv;   // β^T*M^-1
        CoffVec6d acc_bound_coff_A = acc_phi_rank.transpose() * eigen_mat_MInv;

        Eigen::Map<CoffVec6f> h_matrics_vel_cof(m_h_matrics.velBoundCoff[i]);
        h_matrics_vel_cof = vel_bound_coff_A.cast<float>();

        Eigen::Map<CoffVec6f> h_matrics_acc_cof(m_h_matrics.accBoundCoff[i]);
        h_matrics_acc_cof = acc_bound_coff_A.cast<float>();

        // std::cout << "POINT: " << i << std::endl;
        // std::cout << FG_YELLOW "velBoundCoff:\n" FG_DEFAULT << vel_bound_coff_A << std::endl << std::endl << std::endl;
        // std::cout << FG_YELLOW "accBoundCoff:\n" FG_DEFAULT << acc_bound_coff_A << std::endl << std::endl << std::endl;
    }

    RSG_SIM::setMincoTrajParams(m_h_params);
    RSG_SIM::setMtsMatrics(m_h_matrics);
}

} // namespace RSG_SIM