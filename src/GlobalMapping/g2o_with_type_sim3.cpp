#include "GlobalMapping/g2o_with_type_sim3.h"

#include <g2o/core/factory.h>
#include <g2o/stuff/macros.h>


namespace lsd_slam
{
    G2O_USE_TYPE_GROUP(sba);

    G2O_REGISTER_TYPE_GROUP(sim3sophus);

    G2O_REGISTER_TYPE(VERTEX_SIM3_SOPHUS:EXPMAP,c_vertex_sim3);

    G2O_REGISTER_TYPE(EDGE_SIM3_SOPHUS:EXPMAP,c_edge_sim3);

    c_vertex_sim3::c_vertex_sim3(): g2o::BaseVertex<7,Sophus::Sim3d>()
    {
        _marginalized = false;
        _fix_scale = false;
    }

    bool c_vertex_sim3::write(std::ostream& os) const
    {
        return false;
    }

    bool c_vertex_sim3::read(std::istream& is) 
    {
        return false;
    }

    c_edge_sim3::c_edge_sim3():g2o::BaseBinaryEdge<7,Sophus::Sim3d,c_vertex_sim3,c_vertex_sim3>()
    {
        
    }

    bool c_edge_sim3::write(std::ostream& os) const
    {
        return false;
    }
    bool c_edge_sim3::read(std::istream& is) 
    {
        return false;
    }
}
