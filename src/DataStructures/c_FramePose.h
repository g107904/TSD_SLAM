#pragma once
#include "util/Sophus_util.h"
#include "GlobalMapping/g2o_with_type_sim3.h"

namespace lsd_slam
{
    class c_Frame;
    class c_FramePose
    {
        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

            c_FramePose(c_Frame* frame);
            virtual ~c_FramePose();

            c_FramePose* m_tracking_parent;

            Sim3 m_this_to_parent_raw;

            int m_frame_ID;
            c_Frame* m_frame;

	    bool m_is_registered_to_graph;

            bool m_is_in_graph;

            bool m_is_optimized;

            c_vertex_sim3* m_vertex;

            void set_pose_graph_opt_result(Sim3& cam_to_world);
            void apply_pose_graph_opt_result();
            Sim3 get_cam_to_world(int recursion_depth = 0);
	    void invalidateCache();
            
        private:
            
            int m_cached_it;
            static int class_cached_it;

            Sim3 m_cam_to_world;

            Sim3 m_cam_to_world_new;

            bool m_has_unmerged_pose;
    };
}
