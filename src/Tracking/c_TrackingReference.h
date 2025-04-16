#pragma once

#include "util/settings.h"
#include "util/EigenCoreInclude.h"
#include "util/c_read_write_lock.h"

#include <mutex>

namespace lsd_slam
{
    class c_Frame;
    class c_DepthMapPixelHypothesis;
    class c_KeyframeGraph;

    class c_TrackingReference
    {
        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

            c_TrackingReference();
            ~c_TrackingReference();

            void import_frame(c_Frame* frame);

            void make_pointcloud(int level);
            void clear_all();
            void invalidate();

            c_Frame* m_keyframe;
            c_read_write_lock* m_keyframe_lock;
            int m_frame_ID;

            Eigen::Vector3f* m_pos_data[PYRAMID_LEVELS]; // (x,y,z)
            Eigen::Vector2f* m_grad_data[PYRAMID_LEVELS]; // (dx,dy)
            Eigen::Vector2f* m_color_and_var_data[PYRAMID_LEVELS]; //(I,var)

            int* m_point_pos_in_xy_grid[PYRAMID_LEVELS]; // x+y*width
            int m_num_data[PYRAMID_LEVELS];//num of each level

        private:
            int m_w_times_h;
            std::mutex m_access_mutex;
            void release_all();
    };
}
