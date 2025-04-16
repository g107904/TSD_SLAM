#include "DataStructures/c_FramePose.h"
#include "DataStructures/c_Frame.h"

namespace lsd_slam
{
    int c_FramePose::class_cached_it = 0;

    c_FramePose::c_FramePose(c_Frame* frame)
    {
        m_tracking_parent = 0;

        m_this_to_parent_raw = Sim3();

        m_frame_ID = frame->get_id();
        m_frame = frame;

        m_is_registered_to_graph = false;

        m_is_in_graph = false;

        m_is_optimized = false;

        m_vertex = nullptr;

        m_cached_it = -1;

        m_cam_to_world = Sim3();

        m_cam_to_world_new = Sim3();

        m_has_unmerged_pose = false;
    }

    c_FramePose::~c_FramePose()
    {
    }

    void c_FramePose::set_pose_graph_opt_result(Sim3& cam_to_world)
    {
        if(!m_is_in_graph)
            return;

        m_cam_to_world_new = cam_to_world;
        m_has_unmerged_pose = true;
    }

    void c_FramePose::apply_pose_graph_opt_result()
    {
        if(!m_has_unmerged_pose)
            return;
        
        m_cam_to_world = m_cam_to_world_new;
        m_is_optimized = true;
        m_has_unmerged_pose = false;
        class_cached_it++;
    }

    Sim3 c_FramePose::get_cam_to_world(int recursion_depth)
    {
        assert(recursion_depth<5000);

        if(m_is_optimized)
            return m_cam_to_world;

        if(m_cached_it == class_cached_it)
            return m_cam_to_world;
        
        if(m_tracking_parent == nullptr)
            return m_cam_to_world = Sim3();

        m_cached_it = class_cached_it;

        return m_cam_to_world = m_tracking_parent->get_cam_to_world(recursion_depth+1) * m_this_to_parent_raw;
    }
    void c_FramePose::invalidateCache()
    {
	m_cached_it = -1;
    }

}
