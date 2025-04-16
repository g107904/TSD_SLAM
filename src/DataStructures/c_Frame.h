#pragma once
#include "util/Sophus_util.h"
#include "util/settings.h"
#include "DataStructures/c_FramePose.h"
#include "DataStructures/c_FrameMemory.h"
#include <unordered_set>
#include "util/c_read_write_lock.h"

namespace lsd_slam
{
    class c_DepthMapPixelHypothesis;
    class c_TrackingReference;

    class c_Frame
    {
        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
            friend class c_FrameMemory;

            c_Frame(int id,int width,int height,const Eigen::Matrix3f& K,const unsigned char* image);

            c_Frame(int id,int width,int height,const Eigen::Matrix3f& K,const float* image);

            ~c_Frame();

            void set_idepth_and_var(const c_DepthMapPixelHypothesis* new_idepth);

            void set_idepth_and_var_from_ground_truth(const float* depth,float cov_scale = 1.0f);

            void prepare_for_stereo_with(c_Frame* other,Sim3& this_to_other,const Eigen::Matrix3f& K,const int level);

            void set_quick_data(c_TrackingReference* reference);

            void set_reactive_data(c_DepthMapPixelHypothesis* depthmap);

            //access data
            inline int get_id() const;
            inline int get_width(int level = 0) const;
            inline int get_height(int level = 0) const;
            inline const Eigen::Matrix3f& get_K(int level = 0) const;
            inline const Eigen::Matrix3f& get_K_inv(int level = 0) const;
            inline float get_fx(int level = 0) const;
            inline float get_fy(int level = 0) const;
            inline float get_cx(int level = 0) const;
            inline float get_cy(int level = 0) const;
            inline float get_fxi(int level = 0) const;
            inline float get_fyi(int level = 0) const;
            inline float get_cxi(int level = 0) const;
            inline float get_cyi(int level = 0) const;

            inline float* get_image(int level = 0);
            inline const Eigen::Vector4f* get_gradients(int level = 0);
            inline const float* get_max_gradients(int level = 0);
            inline const float* get_idepth(int level = 0);
            inline const float* get_idepth_var(int level = 0);
            
            inline bool has_idepth_been_set() const;

            //reactive
            inline const unsigned char* get_validity_reactive();
            inline const float* get_idepth_reactive();
            inline const float* get_idepth_var_reactive();

            //tracking good 
            inline bool* get_ref_pixel_was_good();
            inline bool* get_ref_pixel_was_good_no_create();
            inline void clear_ref_pixel_was_good();

            inline c_read_write_lock* get_active_lock();

            //pose data
            inline Sim3 get_scaled_cam_to_world(int num = 0);
            inline bool has_tracking_parent();
            inline c_Frame* get_tracking_parent();

            //debug
            inline int get_read_count();
            

            c_FramePose* m_pose;
            Sim3 m_last_constraint_tracked_cam_to_world;

            //neighbors in keyframe graph
            std::unordered_set<c_Frame*,std::hash<c_Frame*>,std::equal_to<c_Frame*>,
                Eigen::aligned_allocator<c_Frame*> > m_neighbors;

            //Multi-Map indicating for which other keyframes with which initialization tracking failed
            std::unordered_multimap<c_Frame*,Sim3,std::hash<c_Frame*>,std::equal_to<c_Frame*>,
                Eigen::aligned_allocator<std::pair<const c_Frame*,Sim3> > >m_tracking_failed;

            bool m_has_depth_been_updated;

            //for quick tracking
            std::mutex m_quick_mutex;
            Eigen::Vector3f* m_quick_pos_data; //x,y,z
            Eigen::Vector2f* m_quick_color_and_var_data;//I,var
            int m_num_of_quick_data;

            //for stereo_with 
            int m_reference_ID;
            int m_reference_level;
            float m_dist_squared;
            Eigen::Matrix3f m_K_other_to_this_R;
            Eigen::Vector3f m_K_other_to_this_t;
            Eigen::Vector3f m_other_to_this_t;
            Eigen::Vector3f m_K_this_to_other_t;
            Eigen::Matrix3f m_this_to_other_R;
            Eigen::Vector3f m_other_to_this_R_row0;
            Eigen::Vector3f m_other_to_this_R_row1;
            Eigen::Vector3f m_other_to_this_R_row2;
            Eigen::Vector3f m_this_to_other_t;

            //statistics data
            float m_initial_tracked_residual;
            int m_num_frames_tracked_on_this;
            int m_num_mapped_on_this;
            int m_num_mapped_on_this_total;
            float m_mean_idepth;
            int m_num_points;
            int m_index_in_keyframes;
            float m_edge_error_sum,m_edges_num;
            int m_num_mappable_pixels;
            float m_mean_information;

        private:
            void initialize(int id,int width,int height,const Eigen::Matrix3f& K);

            void build_image(int level);
            void release_image(int level,bool keep_min_level_data,bool only_clean_flag);

            void build_gradients(int level);
            void release_gradients(int level,bool keep_min_level_data,bool only_clean_flag);

            void build_max_gradients(int level);
            void release_max_gradients(int level,bool keep_min_level_data,bool only_clean_flag);

            void build_idepth_and_var(int level);
            void release_idepth(int level,bool keep_min_level_data,bool only_clean_flag);
            void release_idepth_var(int level,bool keep_min_level_data,bool only_clean_flag);


            bool minimize_memory();

            
            struct c_data
            {
                int m_id;
                int m_width[PYRAMID_LEVELS],m_height[PYRAMID_LEVELS];

                Eigen::Matrix3f m_K[PYRAMID_LEVELS],m_K_inv[PYRAMID_LEVELS];
                float m_fx[PYRAMID_LEVELS],m_fy[PYRAMID_LEVELS],m_cx[PYRAMID_LEVELS],m_cy[PYRAMID_LEVELS];
                float m_fxi[PYRAMID_LEVELS],m_fyi[PYRAMID_LEVELS],m_cxi[PYRAMID_LEVELS],m_cyi[PYRAMID_LEVELS];

                float* m_image[PYRAMID_LEVELS];
                bool m_image_valid[PYRAMID_LEVELS];

                Eigen::Vector4f* m_gradients[PYRAMID_LEVELS];
                bool m_gradients_valid[PYRAMID_LEVELS];

                float* m_max_gradients[PYRAMID_LEVELS];
                bool m_max_gradients_valid[PYRAMID_LEVELS];

                bool m_has_idepth_been_set;

                float* m_idepth[PYRAMID_LEVELS];
                bool m_idepth_valid[PYRAMID_LEVELS];

                float* m_idepth_var[PYRAMID_LEVELS];
                bool m_idepth_var_valid[PYRAMID_LEVELS];

                unsigned char* m_validity_reactive;
                float* m_idepth_reactive;
                float* m_idepth_var_reactive;
                bool m_reactive_data_valid;

                bool* m_ref_pixel_was_good;
            };

            c_data m_data;

            std::mutex m_build_mutex;

            c_read_write_lock m_active_mutex;
            bool m_is_active;

            
    };

    //debug
    inline int c_Frame::get_read_count()
    {
        return m_active_mutex.get_read_count();
    }

    inline int c_Frame::get_id() const
    {
        return m_data.m_id;
    }

    inline int c_Frame::get_width(int level) const
    {
        return m_data.m_width[level];
    }

    inline int c_Frame::get_height(int level) const
    {
        return m_data.m_height[level];
    }

    inline const Eigen::Matrix3f& c_Frame::get_K(int level) const
    {
        return m_data.m_K[level];
    }

    inline const Eigen::Matrix3f& c_Frame::get_K_inv(int level) const
    {
        return m_data.m_K_inv[level];
    }

    inline float c_Frame::get_fx(int level) const
    {
        return m_data.m_fx[level];
    }

    inline float c_Frame::get_fy(int level) const
    {
        return m_data.m_fy[level];
    }

    inline float c_Frame::get_cx(int level) const
    {
        return m_data.m_cx[level];
    }

    inline float c_Frame::get_cy(int level) const
    {
        return m_data.m_cy[level];
    }

    inline float c_Frame::get_fxi(int level) const
    {
        return m_data.m_fxi[level];
    }

    inline float c_Frame::get_fyi(int level) const
    {
        return m_data.m_fyi[level];
    }

    inline float c_Frame::get_cxi(int level) const 
    {
        return m_data.m_cxi[level];
    }

    inline float c_Frame::get_cyi(int level) const
    {
        return m_data.m_cyi[level];
    }

    inline float* c_Frame::get_image(int level)
    {
        if(!m_data.m_image_valid[level])
            build_image(level);
        return m_data.m_image[level];
    }

    inline const Eigen::Vector4f* c_Frame::get_gradients(int level)
    {
        if(!m_data.m_gradients_valid[level])
            build_gradients(level);
        return m_data.m_gradients[level];
    }

    inline const float* c_Frame::get_max_gradients(int level)
    {
        if(!m_data.m_max_gradients_valid[level])
            build_max_gradients(level);
        return m_data.m_max_gradients[level];
    }

    inline const float* c_Frame::get_idepth(int level)
    {
        if(!m_data.m_has_idepth_been_set)
        {
            return nullptr;
        }
        if(!m_data.m_idepth_valid[level])
            build_idepth_and_var(level);
        return m_data.m_idepth[level];
    }

    inline const float* c_Frame::get_idepth_var(int level)
    {
        if(!m_data.m_has_idepth_been_set)
        {
            return nullptr;
        }
        if(!m_data.m_idepth_var_valid[level])
            build_idepth_and_var(level);
        return m_data.m_idepth_var[level];
    }

    inline bool c_Frame::has_idepth_been_set() const 
    {
        return m_data.m_has_idepth_been_set;
    }

    inline const unsigned char* c_Frame::get_validity_reactive()
    {
        if(!m_data.m_reactive_data_valid)
            return 0;
        return m_data.m_validity_reactive;
    }

    inline const float* c_Frame::get_idepth_reactive()
    {
        if(!m_data.m_reactive_data_valid)
            return 0;
        return m_data.m_idepth_reactive;
    }

    inline const float* c_Frame::get_idepth_var_reactive()
    {
        if(!m_data.m_reactive_data_valid)
            return 0;
        return m_data.m_idepth_var_reactive;
    }

    inline bool* c_Frame::get_ref_pixel_was_good()
    {
        if(m_data.m_ref_pixel_was_good == 0)
        {
            std::unique_lock<std::mutex> build_lock(m_build_mutex);

            if(m_data.m_ref_pixel_was_good == 0)
            {
                int width = m_data.m_width[SE3TRACKING_MIN_LEVEL];
                int height = m_data.m_height[SE3TRACKING_MIN_LEVEL];

                m_data.m_ref_pixel_was_good = (bool*) c_FrameMemory::get_instance().dispatch_buffer(sizeof(bool)*width*height);

                memset(m_data.m_ref_pixel_was_good,0xFFFFFFFF,sizeof(bool)*(width*height));

            }
        }
        return m_data.m_ref_pixel_was_good;
    }

    inline bool* c_Frame::get_ref_pixel_was_good_no_create()
    {
        return m_data.m_ref_pixel_was_good;
    }

    inline void c_Frame::clear_ref_pixel_was_good()
    {
        c_FrameMemory::get_instance().reclaim_buffer(reinterpret_cast<float*>(m_data.m_ref_pixel_was_good));
        m_data.m_ref_pixel_was_good = 0;
    }

    inline c_read_write_lock* c_Frame::get_active_lock()
    {
        return c_FrameMemory::get_instance().activate_frame(this);
    }

    inline Sim3 c_Frame::get_scaled_cam_to_world(int num)
    {
        return m_pose->get_cam_to_world();
    }

    inline bool c_Frame::has_tracking_parent()
    {
        return m_pose->m_tracking_parent != nullptr;
    }

    inline c_Frame* c_Frame::get_tracking_parent()
    {
        return m_pose->m_tracking_parent->m_frame;
    }
}
