#ifndef g2o_BA
#define g2o_BA

#include <g2o/core/base_vertex.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_multi_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/core/robust_kernel_impl.h>
#include <iostream>
#include <Eigen/Core>
#include <opencv2/core.hpp>

#include "sophus/se3.hpp"
#include "world_element.h"
#pragma once


class FramePose : public g2o::BaseVertex<6, Sophus::SE3d> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    FramePose() {}

    virtual void setToOriginImpl() override {
        _estimate = Sophus::SE3d();
    }

    virtual void oplusImpl(const double* update) override {
        Eigen::Matrix<double, 6, 1> pose_update;
        for (int i = 0; i < 6; i++)
            pose_update[i] = update[i];
        _estimate = Sophus::SE3d::exp(pose_update) * _estimate;
    }

    Eigen::Vector2d project(const Eigen::Vector3d& point,const Eigen::Matrix3d& K) {
        Eigen::Matrix3d rotation = _estimate.rotationMatrix();
        Eigen::Vector3d trans = _estimate.translation();
        Eigen::Vector3d pc = rotation * point + trans;
        pc = pc / pc[2];
        double r2 = pc.squaredNorm();
        double fx = K(0, 0), fy = K(1, 1), cx = K(0, 2), cy = K(1, 2);
        return Eigen::Vector2d(fx*pc[0]+cx,fy*pc[1]+cy);
    }

    Eigen::Matrix3d x_mul_matrix(Eigen::Vector3d t)
    {
        Eigen::Matrix3d result = Eigen::Matrix3d::Zero();
        result(0, 1) = -t[2]; result(0, 2) = t[1];
        result(1, 0) = t[2]; result(1, 2) = -t[0];
        result(2, 0) = -t[1]; result(2, 1) = t[0];
        return result;
    }

    Eigen::Vector2d project_line(const Eigen::Vector4d& line, const Eigen::Matrix3d& K,const Eigen::Vector4d vertexs)
    {
        Eigen::Matrix3d rotation = _estimate.rotationMatrix();
        Eigen::Vector3d trans = _estimate.translation();
        double fx = K(0, 0), fy = K(1, 1), cx = K(0, 2), cy = K(1, 2);
        Eigen::Matrix3f K_lines = Eigen::Matrix3f::Zero();
        K_lines << fy, 0, 0, 0, fx, 0, -fy * cx, -fx * cy, fx* fy;
        world_line tmp;
        tmp.pos = line.cast<float>();
        Eigen::Matrix<float, 6, 1> tmp_plucker = tmp.to_plucker();

        Eigen::Matrix3f tmp_mul = (x_mul_matrix(trans) * rotation).cast<float>();
        Eigen::Matrix<float, 6, 6> tmp_matrix = Eigen::Matrix<float, 6, 6>::Zero();
        for(int i = 0;i < 2;i++)
            for (int j = 0; j < 2; j++)
            {
                tmp_matrix(i, j) = rotation(i, j);
                tmp_matrix(i + 3, j + 3) = rotation(i, j);
                tmp_matrix(i, j + 3) = tmp_mul(i, j);
            }
        tmp_plucker = tmp_matrix * tmp_plucker;



        Eigen::Vector3f n;n << tmp_plucker[0], tmp_plucker[1], tmp_plucker[2];
        Eigen::Vector3f v;v << tmp_plucker[3], tmp_plucker[4], tmp_plucker[5];


        Eigen::Vector3f l = K_lines * n;
        Eigen::Vector2d e1; e1 << vertexs[0], vertexs[1];
        Eigen::Vector2d e2; e2 << vertexs[2], vertexs[3];
        Eigen::Vector3d eq1;eq1 << e1[0], e1[1], 1;
        Eigen::Vector3d eq2; eq2 << e2[0], e2[1], 1;
        eq1 = K.inverse() * eq1; eq2 = K.inverse() * eq2;
        Eigen::Vector2d result;
        double div = sqrt(l[0] * l[0] + l[1] * l[1]);
        result[0] = l.cast<double>().dot(eq1);
        result[1] = l.cast<double>().dot(eq2);
        return result;

    }

    bool read(std::istream& in) { return false; }

    bool write(std::ostream& out) const { return false; }
};

class VertexPoint : public g2o::BaseVertex<3, Eigen::Vector3d> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    VertexPoint() {}

    virtual void setToOriginImpl() override {
        _estimate = Eigen::Vector3d(0, 0, 0);  
    }

    virtual void oplusImpl(const double* update) override {
        _estimate += Eigen::Vector3d(update[0], update[1], update[2]);
    }

    bool read(std::istream& in) { return false; }

    bool write(std::ostream& out) const { return false; }
};

class EdgeProjection :
    public g2o::BaseBinaryEdge<2, Eigen::Vector2d,FramePose, VertexPoint> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    Eigen::Matrix3d K;
    EdgeProjection(Eigen::Matrix3d& t_K)
    {
        K = t_K;
    }

    virtual void computeError() override {
        auto v0 = (FramePose*)_vertices[0];
        auto v1 = (VertexPoint*)_vertices[1];
        auto proj = v0->project(v1->estimate(),K);
        _error = proj - _measurement;
    }

    virtual void linearizeOplus() override {
        auto v0 = (FramePose*)_vertices[0];
        Sophus::SE3d pose = v0->estimate();
        auto fx = K(0, 0);
        auto fy = K(1,1);
        auto cx = K(0,2);
        auto cy = K(1,2);
        auto v1 = (VertexPoint*)_vertices[1];
        Eigen::Vector3d Pc = pose * v1->estimate();
        auto X = Pc[0];
        auto Y = Pc[1];
        auto Z = Pc[2];
        auto Z2 = Z * Z;
        _jacobianOplusXi <<
            -fx / Z, 0, fx* X / Z2, fx* X* Y / Z2, -fx - fx * X * X / Z2, fx* Y / Z,
            0, -fy / Z, fy* Y / (Z * Z), fy + fy * Y * Y / Z2, -fy * X * Y / Z2, -fy * X / Z;
        Eigen::Matrix3d rot = pose.rotationMatrix();
        Eigen::Vector3d trans = pose.translation();
        Eigen::Vector3d r1 = rot.row(0);
        Eigen::Vector3d r2 = rot.row(1);
        Eigen::Vector3d r3 = rot.row(2);
        Eigen::Vector3d partialx = r1 * trans[2] - r3 * trans[0];
        Eigen::Vector3d partialy = r2 * trans[2] - r3 * trans[1];
        partialx = -fx * partialx / Z2; partialy = -fy * partialy / Z2;
        _jacobianOplusXj << 
            partialx.transpose(),partialy.transpose()
            ;
    }
    

    // use numeric derivatives
    bool read(std::istream& in) { return false; }

    bool write(std::ostream& out) const { return false; }

};

class EdgePointLine : public g2o::BaseVertex<4, Eigen::Vector4d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    EdgePointLine() {}

    virtual void setToOriginImpl() override {
        _estimate = Eigen::Vector4d(0,0, 0, 0);
    }

    virtual void oplusImpl(const double* update) override {
        _estimate += Eigen::Vector4d(update[0], update[1], update[2],update[3]);
    }

    bool read(std::istream& in) { return false; }

    bool write(std::ostream& out) const { return false; }
};

class EdgeLineProjection : public g2o::BaseBinaryEdge<2, Eigen::Vector4d, FramePose, EdgePointLine> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    Eigen::Matrix3d K;
    EdgeLineProjection(Eigen::Matrix3d& t_K)
    {
        K = t_K;
    }

    virtual void computeError() override {
        auto v0 = (FramePose*)_vertices[0];
        auto v1 = (EdgePointLine*)_vertices[1];
        if (v1->estimate()[3] == 0)
        {
            Eigen::Vector4d tmp = v1->estimate();
            auto proj = v0->project(Eigen::Vector3d(tmp[0],tmp[1],tmp[2]), K);
            _error = proj - Eigen::Vector2d(_measurement[0],_measurement[1]);
        }
        else
        {
            _error = v0->project_line(v1->estimate(), K, _measurement);
        }
    }

    virtual void linearizeOplus() override {
        auto v0 = (FramePose*)_vertices[0];
        Sophus::SE3d pose = v0->estimate();
        auto fx = K(0, 0);
        auto fy = K(1, 1);
        auto cx = K(0, 2);
        auto cy = K(1, 2);
        auto v1 = (EdgePointLine*)_vertices[1];
        if (v1->estimate()[3] == 0)
        {
            Eigen::Vector4d tmp = v1->estimate();
            Eigen::Vector3d Pc = pose * Eigen::Vector3d(tmp[0], tmp[1], tmp[2]);

            auto X = Pc[0];
            auto Y = Pc[1];
            auto Z = Pc[2];
            auto Z2 = Z * Z;
            _jacobianOplusXi <<
                -fx / Z, 0, fx* X / Z2, fx* X* Y / Z2, -fx - fx * X * X / Z2, fx* Y / Z,
                0, -fy / Z, fy* Y / (Z * Z), fy + fy * Y * Y / Z2, -fy * X * Y / Z2, -fy * X / Z;
            Eigen::Matrix3d rot = pose.rotationMatrix();
            Eigen::Vector3d trans = pose.translation();
            Eigen::Vector3d r1 = rot.row(0);
            Eigen::Vector3d r2 = rot.row(1);
            Eigen::Vector3d r3 = rot.row(2);
            Eigen::Vector3d partialx = r1 * trans[2] - r3 * trans[0];
            Eigen::Vector3d partialy = r2 * trans[2] - r3 * trans[1];
            partialx = -fx * partialx / Z2; partialy = -fy * partialy / Z2;
            _jacobianOplusXj <<
                partialx.transpose(), partialy.transpose()
                ;
        }
        else
        {
            Eigen::Matrix3d rot = pose.inverse().rotationMatrix().transpose(); 
            Eigen::Vector3d trans = pose.inverse().translation();
            Eigen::Vector4d line = v1->estimate();

            Eigen::Matrix<double, 2, 3> matrix_rL_div_l;
            double fx = K(0, 0), fy = K(1, 1), cx = K(0, 2), cy = K(1, 2);
            Eigen::Matrix3f K_lines = Eigen::Matrix3f::Zero();
            K_lines << fy, 0, 0, 0, fx, 0, -fy * cx, -fx * cy, fx* fy;
            world_line tmp;
            tmp.pos = line.cast<float>();
            Eigen::Matrix<float, 6, 1> tmp_plucker = tmp.to_plucker();
            Eigen::Vector3d n; n << tmp_plucker[0], tmp_plucker[1], tmp_plucker[2];
            Eigen::Vector3d v; v << tmp_plucker[3], tmp_plucker[4], tmp_plucker[5];
            Eigen::Vector3d l = K_lines.cast<double>() * n;

            double uepc = _measurement[0], vepc = _measurement[1], ueqc = _measurement[2], veqc = _measurement[3];
            double l_sqrt = sqrt(l[0] * l[0] + l[1] * l[1]);
            double l_sqrt_3 = l_sqrt * l_sqrt * l_sqrt; l_sqrt_3 = 1.0 / l_sqrt_3;
            l_sqrt = 1.0 / l_sqrt;
            matrix_rL_div_l << (uepc*l[1]*l[1] - vepc*l[0]*l[1] - l[2]*l[0])*l_sqrt_3,(vepc*l[0]*l[0] - l[1]*l[2] - uepc*l[0]*l[1])*l_sqrt_3,l_sqrt,
                (ueqc * l[1] * l[1] - veqc * l[0] * l[1])*l_sqrt_3,(veqc * l[0] * l[0] - uepc * l[0] * l[1])*l_sqrt_3,l_sqrt;

            Eigen::Matrix<double, 3, 6> matrix_l_div_L = Eigen::Matrix<double,3,6>::Zero();
            matrix_l_div_L << K_lines.cast<double>(), Eigen::Matrix3d::Zero();

            Eigen::Matrix<double, 6, 6> matrix_L_div_lw = Eigen::Matrix<double, 6, 6>();
            Eigen::Matrix3d tmp_mul = rot * v0->x_mul_matrix(trans);
            for(int i = 0;i < 2;i++)
                for (int j = 0; j < 2; j++)
                {
                    matrix_L_div_lw(i, j) = rot(i, j);
                    matrix_L_div_lw(i + 3, j + 3) = rot(i, j);
                    matrix_L_div_lw(i, j + 3) = tmp_mul(i, j);
                }

            Eigen::Matrix<double, 6, 6> matrix_L_div_pose = Eigen::Matrix<double, 6, 6>::Zero();
            Eigen::Vector3d matrix_vector_0 = rot * (n+v0->x_mul_matrix(trans)*v);
            Eigen::Matrix3d matrix0 = v0->x_mul_matrix(matrix_vector_0);
            Eigen::Vector3d matrix_vector_1 = rot * v;
            Eigen::Matrix3d matrix1 = v0->x_mul_matrix(matrix_vector_1);
            Eigen::Matrix3d matrix2 = -rot * v0->x_mul_matrix(v);
            for(int i = 0;i < 2;i++)
                for (int j = 0; j < 2; j++)
                {
                    matrix_L_div_pose(i, j) = matrix0(i, j);
                    matrix_L_div_pose(i + 3, j) = matrix1(i, j);
                    matrix_L_div_pose(i, j + 3) = matrix2(i, j);
                }

            double n_norm = n.norm();
            double w1 = n_norm / sqrt(n_norm * n_norm + 1), w2 = 1 / sqrt(n_norm * n_norm + 1);
            Eigen::Vector3d u1 = n / n_norm, u2 = v;
            Eigen::Vector3d u3 = n.cross(v); u3.normalize();

            Eigen::Matrix<double, 6, 4> matrix_lw_div_o = Eigen::Matrix<double, 6, 4>::Zero();
            Eigen::Matrix<double, 6, 1> vectors[4];
            vectors[0] << Eigen::Vector3d::Zero(), -w2 * u3;
            vectors[1] << -w1 * u3, Eigen::Vector3d::Zero();
            vectors[2] << w1 * u2, -w2 * u1;
            vectors[3] << -w2 * u1, w1* u2;
            matrix_lw_div_o << vectors[0], vectors[1], vectors[2], vectors[3];

            _jacobianOplusXi = matrix_rL_div_l * matrix_l_div_L * matrix_L_div_pose;
            
            _jacobianOplusXj = matrix_rL_div_l * matrix_l_div_L * matrix_L_div_lw * matrix_lw_div_o;
        }
    }


    // use numeric derivatives
    bool read(std::istream& in) { return false; }

    bool write(std::ostream& out) const { return false; }

};



void SolveBA(
    Eigen::Matrix3d K,
    std::vector<world_point>& points,
    std::vector<std::vector<cv::KeyPoint>>& frame_kp,
    std::vector<Sophus::SE3f>& frame_pose
)
{
    int n_all_point = points.size();
    int key_n = frame_pose.size();

    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> BlockSolverType;
    typedef g2o::LinearSolverCSparse<BlockSolverType::PoseMatrixType> LinearSolverType;
    auto solver = new LinearSolverType();
    g2o::SparseOptimizer optimizer; 
    auto blocksolver = new BlockSolverType(solver);
    g2o::OptimizationAlgorithmLevenberg* algorithm = new g2o::OptimizationAlgorithmLevenberg(blocksolver);
    optimizer.setAlgorithm(algorithm);  
    optimizer.setVerbose(true);  

    // vertex
    std::vector<FramePose*> vertex_pose;
    std::vector<VertexPoint*> vertex_points;
    for (int i = 0; i < key_n; i++)
    {
        FramePose* v = new FramePose();
        v->setId(i);
        v->setEstimate(frame_pose[i].inverse().cast<double>());
        if (i == 0)
            v->setFixed(true);
        optimizer.addVertex(v);
        vertex_pose.push_back(v);
    }


    for (int i = 0; i < n_all_point; i++)
    {
        VertexPoint* v = new VertexPoint();
        v->setId(i+key_n);
        v->setEstimate(points[i].pos.cast<double>());
        optimizer.addVertex(v);
        vertex_points.push_back(v);
    }


    for (int i = 0; i < n_all_point; i++)
    {
        for (int j = 0; j < points[i].frame_pos.size(); j++)
        {
            EdgeProjection* edge = new EdgeProjection(K);
            edge->setVertex(0, vertex_pose[j]);
            edge->setVertex(1, vertex_points[i]);
            cv::KeyPoint kp = frame_kp[j][points[i].frame_pos[j]];
            edge->setMeasurement(Eigen::Vector2d(kp.pt.x, kp.pt.y));
            edge->setInformation(Eigen::Matrix2d::Identity());
            edge->setRobustKernel(new g2o::RobustKernelHuber());
            optimizer.addEdge(edge);
        }
    }

    optimizer.initializeOptimization();
    optimizer.optimize(100);

    for (int i = 0; i < key_n; i++)
    {
        auto estimate = vertex_pose[i]->estimate();
        frame_pose[i] = estimate.inverse().cast<float>();
    }
}
#endif