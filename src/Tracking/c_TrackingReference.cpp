#include "Tracking/c_TrackingReference.h"
#include "DataStructures/c_Frame.h"
#include "DepthEstimation/c_DepthMapPixelHypothesis.h"
#include "GlobalMapping/c_KeyFrameGraph.h"
#include "util/interpolation.h"


namespace lsd_slam
{
    c_TrackingReference::c_TrackingReference()
    {
        m_frame_ID = -1;
        m_keyframe = 0;
        m_w_times_h = 0;
        for(int level = 0;level < PYRAMID_LEVELS;level++)
        {
            m_pos_data[level] = nullptr;
            m_grad_data[level] = nullptr;
            m_color_and_var_data[level] = nullptr;
            m_point_pos_in_xy_grid[level] = nullptr;
            m_num_data[level] = 0;
        }
    }

    c_TrackingReference::~c_TrackingReference()
    {
        std::unique_lock<std::mutex> lock(m_access_mutex);
        invalidate();
        release_all();
    }

    void c_TrackingReference::import_frame(c_Frame* frame)
    {
        std::unique_lock<std::mutex> lock(m_access_mutex);
        m_keyframe_lock = frame->get_active_lock();
        m_keyframe = frame;
        m_frame_ID = frame->get_id();

        if(frame->get_width(0) * frame->get_height(0) != m_w_times_h)
        {
            release_all();
            m_w_times_h = frame->get_width(0) * frame->get_height(0);
        }

        clear_all();
        lock.unlock();
    }

    void c_TrackingReference::make_pointcloud(int level)
    {
        assert(m_keyframe != 0);

        std::unique_lock<std::mutex> lock(m_access_mutex);

        if(m_num_data[level] > 0) // already exists
            return;
        
        int w = m_keyframe->get_width(level);
        int h = m_keyframe->get_height(level);

        float fxi_level = m_keyframe->get_fxi(level);
        float fyi_level = m_keyframe->get_fyi(level);
        float cxi_level = m_keyframe->get_cxi(level);
        float cyi_level = m_keyframe->get_cyi(level);

        const float* idepth_level = m_keyframe->get_idepth(level);
        const float* idepth_var_level = m_keyframe->get_idepth_var(level);
        const float* color_level = m_keyframe->get_image(level);
        const Eigen::Vector4f* grad_level = m_keyframe->get_gradients(level);

        if(m_pos_data[level] == nullptr)
            m_pos_data[level] = new Eigen::Vector3f[w*h];
        if(m_point_pos_in_xy_grid[level] == nullptr)
            m_point_pos_in_xy_grid[level] = (int*) Eigen::internal::aligned_malloc(w*h*sizeof(int));
        if(m_grad_data[level] == nullptr)
            m_grad_data[level] = new Eigen::Vector2f[w*h];
        if(m_color_and_var_data[level] == nullptr)
            m_color_and_var_data[level] = new Eigen::Vector2f[w*h];
        
        Eigen::Vector3f* m_pos_data_it = m_pos_data[level];
        int* m_point_pos_in_xy_grid_it = m_point_pos_in_xy_grid[level];
        Eigen::Vector2f* m_grad_data_it = m_grad_data[level];
        Eigen::Vector2f* m_color_and_var_data_it = m_color_and_var_data[level];
	
	int num = 0;

        for(int x = 1;x < w-1;x++)
            for(int y = 1;y < h-1;y++)
            {
                int index = x + y * w;

                if(idepth_var_level[index] <= 0 || idepth_level[index] == 0)
                    continue;

		num++;
                
                *m_pos_data_it = (1.0f / idepth_level[index]) * Eigen::Vector3f(fxi_level*x+cxi_level,fyi_level*y+cyi_level,1);
                *m_point_pos_in_xy_grid_it = index;
                *m_grad_data_it = grad_level[index].head<2>();
                *m_color_and_var_data_it = Eigen::Vector2f(color_level[index],idepth_var_level[index]);


                m_pos_data_it++;
                m_point_pos_in_xy_grid_it++;
                m_grad_data_it++;
                m_color_and_var_data_it++;
            }

        m_num_data[level] = m_pos_data_it - m_pos_data[level];
    }

    void c_TrackingReference::clear_all()
    {
        for(int level = 0;level < PYRAMID_LEVELS;level++)
            m_num_data[level] = 0;
    }

    void c_TrackingReference::invalidate()
    {
        if(m_keyframe != 0)
            m_keyframe_lock->unlock_shared();
        m_keyframe = 0;
    }

    void c_TrackingReference::release_all()
    {
        for(int level = 0;level < PYRAMID_LEVELS;level++)
        {
            if(m_pos_data[level] != nullptr)
                delete[] m_pos_data[level];
            if(m_grad_data[level] != nullptr)
                delete[] m_grad_data[level];
            if(m_color_and_var_data[level] != nullptr)
                delete[] m_color_and_var_data[level];
            if(m_point_pos_in_xy_grid[level] != nullptr)
                Eigen::internal::aligned_free((void*)m_point_pos_in_xy_grid[level]);
            m_num_data[level] = 0;
        }

        m_w_times_h = 0;
    }
}
