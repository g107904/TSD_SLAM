#pragma once
#include "util/Sophus_util.h"

#include "g2o/core/base_vertex.h"
#include "g2o/core/base_binary_edge.h"
#include "g2o/types/sba/types_six_dof_expmap.h"

namespace lsd_slam
{
    class c_vertex_sim3 : public g2o::BaseVertex<7,Sophus::Sim3d>
    {
        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
            c_vertex_sim3();
  
            virtual bool read(std::istream& is);
            virtual bool write(std::ostream& os) const;
            
            virtual void setToOriginImpl()
            {
                _estimate = Sophus::Sim3d();
            }

            virtual void oplusImpl(const double* update_)
            {
                Eigen::Map<Eigen::Matrix<double,7,1> > update(const_cast<double*>(update_));

                if(_fix_scale)
                    update[6] = 0;
                setEstimate(Sophus::Sim3d::exp(update) * estimate());
            }

            bool _fix_scale;
    };

    class c_edge_sim3 : public g2o::BaseBinaryEdge<7,Sophus::Sim3d,c_vertex_sim3,c_vertex_sim3>
    {
        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
            c_edge_sim3();
            virtual bool read(std::istream& is);
            virtual bool write(std::ostream& os) const;

            void computeError()
            {
                const c_vertex_sim3* v_from = static_cast<const c_vertex_sim3*>(_vertices[0]);
                const c_vertex_sim3* v_to = static_cast<const c_vertex_sim3*>(_vertices[1]);

                Sophus::Sim3d error = v_from->estimate().inverse() * v_to->estimate() * _inverseMeasurement;
                _error = error.log();
            }

            void linearizeOplus()
            {
                const c_vertex_sim3* v_from = static_cast<const c_vertex_sim3*>(_vertices[0]);
                
                _jacobianOplusXj = v_from->estimate().inverse().Adj();
                _jacobianOplusXi = -_jacobianOplusXj;
            }

            virtual void setMeasurement(const Sophus::Sim3d& m)
            {
                _measurement = m;
                _inverseMeasurement = m.inverse();
            }

            virtual bool setMeasurementData(const double* m)
            {
                Eigen::Map<const g2o::Vector7d> v(m);
                setMeasurement(Sophus::Sim3d::exp(v));
                return true;
            }

            virtual bool setMeasurementFromState()
            {
                const c_vertex_sim3* v_from = static_cast<const c_vertex_sim3*>(_vertices[0]);
                const c_vertex_sim3* v_to = static_cast<const c_vertex_sim3*>(_vertices[1]);
                Sophus::Sim3d delta = v_from->estimate().inverse() * v_to->estimate();
                setMeasurement(delta);
                return true;
            }

            virtual double initialEstimatePossible(const g2o::OptimizableGraph::VertexSet&,g2o::OptimizableGraph::Vertex*)
            {
                return 1;
            }

            virtual void initialEstimate(const g2o::OptimizableGraph::VertexSet& from,g2o::OptimizableGraph::Vertex*)
            {
                c_vertex_sim3* v_from = static_cast<c_vertex_sim3*>(_vertices[0]);
                c_vertex_sim3* v_to = static_cast<c_vertex_sim3*>(_vertices[1]);

                if(from.count(v_from) > 0)
                {
                    v_to->setEstimate(v_from->estimate() * _measurement);
                }
                else
                {
                    v_from->setEstimate(v_to->estimate() * _inverseMeasurement);
                }
            }
            protected:
                Sophus::Sim3d _inverseMeasurement;

    };
}
