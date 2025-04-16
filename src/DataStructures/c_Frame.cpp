#include "DataStructures/c_Frame.h"
#include "DataStructures/c_FrameMemory.h"
#include "DepthEstimation/c_DepthMapPixelHypothesis.h"
#include "Tracking/c_TrackingReference.h"
#include <cmath>
#include <cfloat>


namespace lsd_slam
{
    c_Frame::c_Frame(int id,int width,int height,const Eigen::Matrix3f& K,const unsigned char* image)
    {
        initialize(id,width,height,K);

        m_data.m_image[0] = (float*)c_FrameMemory::get_instance().dispatch_buffer(sizeof(float) * m_data.m_width[0] * m_data.m_height[0]);

        float* it_end = m_data.m_image[0] + m_data.m_width[0] * m_data.m_height[0];

        for(float* it = m_data.m_image[0];it != it_end;it++)
        {
            *it = *image;
            image++;
        }

        m_data.m_image_valid[0] = true;
    }

    c_Frame::c_Frame(int id,int width,int height,const Eigen::Matrix3f& K,const float* image)
    {
        initialize(id,width,height,K);

        m_data.m_image[0] = (float*)c_FrameMemory::get_instance().dispatch_buffer(sizeof(float) * m_data.m_width[0] * m_data.m_height[0]);

        memcpy(m_data.m_image[0],image,sizeof(float) * m_data.m_width[0] * m_data.m_height[0]);

        m_data.m_image_valid[0] = true;
    }

    c_Frame::~c_Frame()
    {
        c_FrameMemory::get_instance().deactivate_frame(this);

        if(!m_pose->m_is_registered_to_graph)
            delete m_pose;
        else
            m_pose->m_frame = 0;
        
        for(int level = 0;level < PYRAMID_LEVELS;level++)
        {
            c_FrameMemory::get_instance().reclaim_buffer(m_data.m_image[level]);
            c_FrameMemory::get_instance().reclaim_buffer(reinterpret_cast<float*>(m_data.m_gradients[level]));
            c_FrameMemory::get_instance().reclaim_buffer(m_data.m_max_gradients[level]);
            c_FrameMemory::get_instance().reclaim_buffer(m_data.m_idepth[level]);
            c_FrameMemory::get_instance().reclaim_buffer(m_data.m_idepth_var[level]);

        }

        c_FrameMemory::get_instance().reclaim_buffer((float*) m_data.m_validity_reactive);
        c_FrameMemory::get_instance().reclaim_buffer(m_data.m_idepth_reactive);
        c_FrameMemory::get_instance().reclaim_buffer(m_data.m_idepth_var_reactive);

        if(m_quick_pos_data != 0)
            delete m_quick_pos_data;
        if(m_quick_color_and_var_data != 0)
            delete m_quick_color_and_var_data;
    }

    void c_Frame::set_idepth_and_var(const c_DepthMapPixelHypothesis* new_idepth)
    {
        c_read_write_lock* lock = get_active_lock();
        std::unique_lock<std::mutex> build_lock(m_build_mutex);

        if(m_data.m_idepth[0] == 0)
            m_data.m_idepth[0] = (float*)c_FrameMemory::get_instance().dispatch_buffer(sizeof(float) * m_data.m_width[0] * m_data.m_height[0]);
        if(m_data.m_idepth_var[0] == 0)
            m_data.m_idepth_var[0] = (float*)c_FrameMemory::get_instance().dispatch_buffer(sizeof(float) * m_data.m_width[0] * m_data.m_height[0]);
        
        float sum_idepth = 0;
        int num_idepth = 0;

        float* idepth_it = m_data.m_idepth[0];
        float* idepth_var_it = m_data.m_idepth_var[0];
        float* idepth_it_end = idepth_it + (m_data.m_width[0] * m_data.m_height[0]);

        for(;idepth_it < idepth_it_end;idepth_it++,idepth_var_it++,new_idepth++)
        {
            if(new_idepth->m_isValid && new_idepth->m_idepth_smoothed >= -0.05)
            {
                *idepth_it = new_idepth->m_idepth_smoothed;
                *idepth_var_it = new_idepth->m_idepth_var_smoothed;

                num_idepth++;
                sum_idepth += new_idepth->m_idepth_smoothed;
            }
            else
            {
                *idepth_it = -1;
                *idepth_var_it = -1;
            }
        }

        m_mean_idepth = sum_idepth / num_idepth;
        m_num_points = num_idepth;

        m_data.m_idepth_valid[0] = true;
        m_data.m_idepth_var_valid[0] = true;

        release_idepth(PYRAMID_LEVELS-1,true,true);
        release_idepth_var(PYRAMID_LEVELS-1,true,true);

        m_data.m_has_idepth_been_set = true;
        m_has_depth_been_updated = true;

        lock->unlock_shared();
    }

    void c_Frame::set_idepth_and_var_from_ground_truth(const float* depth,float cov_scale)
    {
        c_read_write_lock* lock = get_active_lock();
        const float* max_gradient_it = get_max_gradients(0);

        std::unique_lock<std::mutex> build_lock(m_build_mutex);

        if(m_data.m_idepth[0] == 0)
            m_data.m_idepth[0] = (float*)c_FrameMemory::get_instance().dispatch_buffer(sizeof(float) * m_data.m_width[0] * m_data.m_height[0]);
        if(m_data.m_idepth_var[0] == 0)
            m_data.m_idepth_var[0] = (float*)c_FrameMemory::get_instance().dispatch_buffer(sizeof(float) * m_data.m_width[0] * m_data.m_height[0]);
        
        float* idepth_it = m_data.m_idepth[0];
        float* idepth_var_it = m_data.m_idepth_var[0];

        int width = m_data.m_width[0];
        int height = m_data.m_height[0];

        for(int y = 0;y < height;y++)
            for(int x = 0;x < width;x++)
            {
                if(x > 0 && x < width - 1 
                    && y > 0 && y < height - 1 
                        && max_gradient_it[x+y*width] >= MIN_ABS_GRAD_CREATE 
                            && !std::isnan(*depth) && *depth > 0)
                            {
                                *idepth_it = 1.0f / (*depth);
                                *idepth_var_it = VAR_GT_INIT_INITIAL * cov_scale;
                            }
                else
                {
                    *idepth_it = -1;
                    *idepth_var_it = -1;
                }

                ++depth;
                ++idepth_it;
                ++idepth_var_it;
            }
        
        m_data.m_idepth_valid[0] = true;
        m_data.m_idepth_var_valid[0] = true;

        release_idepth(PYRAMID_LEVELS-1,true,true);
        release_idepth_var(PYRAMID_LEVELS-1,true,true);

        m_data.m_has_idepth_been_set = true;

        lock->unlock_shared();
    }

    void c_Frame::prepare_for_stereo_with(c_Frame* other,Sim3& this_to_other,const Eigen::Matrix3f& K,const int level)
    {
        Sim3 other_to_this = this_to_other.inverse();

        m_K_other_to_this_R = K * other_to_this.rotationMatrix().cast<float>() * other_to_this.scale();
        m_other_to_this_t = other_to_this.translation().cast<float>();
        m_K_other_to_this_t = K * m_other_to_this_t;

        m_this_to_other_t = this_to_other.translation().cast<float>();
        m_K_this_to_other_t = K * m_this_to_other_t;
        m_this_to_other_R = this_to_other.rotationMatrix().cast<float>() * this_to_other.scale();

        m_other_to_this_R_row0 = m_this_to_other_R.col(0);
        m_other_to_this_R_row1 = m_this_to_other_R.col(1);
        m_other_to_this_R_row2 = m_this_to_other_R.col(2);

        m_dist_squared = other_to_this.translation().dot(other_to_this.translation());

        m_reference_ID = other->get_id();
        m_reference_level = level;
    }

    void c_Frame::set_quick_data(c_TrackingReference* reference)
    {
        assert(reference->m_frame_ID == get_id());

        reference->make_pointcloud(QUICK_KF_CHECK_LVL);

        m_quick_mutex.lock();

        if(m_quick_color_and_var_data != 0)
            delete m_quick_color_and_var_data;
        if(m_quick_pos_data != 0)
            delete m_quick_pos_data;
        
        m_num_of_quick_data = reference->m_num_data[QUICK_KF_CHECK_LVL];

        m_quick_color_and_var_data = new Eigen::Vector2f[m_num_of_quick_data];
        m_quick_pos_data = new Eigen::Vector3f[m_num_of_quick_data];

        memcpy(m_quick_color_and_var_data,reference->m_color_and_var_data[QUICK_KF_CHECK_LVL],sizeof(Eigen::Vector2f) * m_num_of_quick_data);
        memcpy(m_quick_pos_data,reference->m_pos_data[QUICK_KF_CHECK_LVL],sizeof(Eigen::Vector3f) * m_num_of_quick_data);

        m_quick_mutex.unlock();
    }

    void c_Frame::set_reactive_data(c_DepthMapPixelHypothesis* depthmap)
    {
        c_read_write_lock* lock = get_active_lock();

        if(m_data.m_validity_reactive == 0)
            m_data.m_validity_reactive = (unsigned char*) c_FrameMemory::get_instance().dispatch_buffer(sizeof(unsigned char) * m_data.m_width[0] * m_data.m_height[0]);
        if(m_data.m_idepth_reactive == 0)
            m_data.m_idepth_reactive = (float*) c_FrameMemory::get_instance().dispatch_buffer(sizeof(float) * m_data.m_width[0] * m_data.m_height[0]);
        if(m_data.m_idepth_var_reactive == 0)
            m_data.m_idepth_var_reactive = (float*) c_FrameMemory::get_instance().dispatch_buffer(sizeof(float) * m_data.m_width[0] * m_data.m_height[0]);
        
        float* idepth_it = m_data.m_idepth_reactive;
        float* idepth_it_end = idepth_it + (m_data.m_width[0] * m_data.m_height[0]);
        float* idepth_var_it = m_data.m_idepth_var_reactive;
        unsigned char* validity_it = m_data.m_validity_reactive;

        for(;idepth_it < idepth_it_end; ++idepth_it,++idepth_var_it,++validity_it,++depthmap)
        {
            if(depthmap->m_isValid)
            {
                *idepth_it = depthmap->m_idepth;
                *idepth_var_it = depthmap->m_idepth_var;
                *validity_it = depthmap->m_validity_counter;
            }
            else if(depthmap->m_blacklisted < MIN_BLACKLIST)
            {
                *idepth_var_it = -2;
            }
            else
            {
                *idepth_var_it = -1;
            }
        } 

        m_data.m_reactive_data_valid = true;

        lock->unlock_shared();
    }

    void c_Frame::initialize(int id,int width,int height,const Eigen::Matrix3f& K)
    {
        m_data.m_id = id;

        m_pose = new c_FramePose(this);

        m_data.m_K[0] = K;
        m_data.m_fx[0] = K(0,0);
        m_data.m_fy[0] = K(1,1);
        m_data.m_cx[0] = K(0,2);
        m_data.m_cy[0] = K(1,2);

        m_data.m_K_inv[0] = K.inverse();
        m_data.m_fxi[0] = m_data.m_K_inv[0](0,0);
        m_data.m_fyi[0] = m_data.m_K_inv[0](1,1);
        m_data.m_cxi[0] = m_data.m_K_inv[0](0,2);
        m_data.m_cyi[0] = m_data.m_K_inv[0](1,2);

        m_data.m_has_idepth_been_set = false;
        m_has_depth_been_updated = false;

        m_reference_ID = -1;
        m_reference_level = -1;

        m_num_mappable_pixels = -1;

        for(int level = 0;level < PYRAMID_LEVELS;++level)
        {
            m_data.m_width[level] = width >> level;
            m_data.m_height[level] = height >> level;

            m_data.m_image_valid[level] = false;
            m_data.m_gradients_valid[level] = false;
            m_data.m_max_gradients_valid[level] = false;
            m_data.m_idepth_valid[level] = false;
            m_data.m_idepth_var_valid[level] = false;

            m_data.m_image[level] = 0;
            m_data.m_gradients[level] = 0;
            m_data.m_max_gradients[level] = 0;
            m_data.m_idepth[level] = 0;
            m_data.m_idepth_var[level] = 0;
            m_data.m_reactive_data_valid = false;

            if(level > 0)
            {
                m_data.m_fx[level] = m_data.m_fx[level-1] * 0.5;
                m_data.m_fy[level] = m_data.m_fy[level-1] * 0.5;
                m_data.m_cx[level] = (m_data.m_cx[0] + 0.5) / ((int) 1<< level ) - 0.5;
                m_data.m_cy[level] = (m_data.m_cy[0] + 0.5) / ((int) 1<< level ) - 0.5;
	
		m_data.m_K[level].setZero();

                m_data.m_K[level](0,0) = m_data.m_fx[level];
                m_data.m_K[level](1,1) = m_data.m_fy[level];
                m_data.m_K[level](0,2) = m_data.m_cx[level];
                m_data.m_K[level](1,2) = m_data.m_cy[level];
                m_data.m_K[level](2,2) = 1.0;

                m_data.m_K_inv[level] = (m_data.m_K[level]).inverse();

                m_data.m_fxi[level] = m_data.m_K_inv[level](0,0);
                m_data.m_fyi[level] = m_data.m_K_inv[level](1,1);
                m_data.m_cxi[level] = m_data.m_K_inv[level](0,2);
                m_data.m_cyi[level] = m_data.m_K_inv[level](1,2);
            }
        }

        m_data.m_validity_reactive = 0;
        m_data.m_idepth_reactive = 0;
        m_data.m_idepth_var_reactive = 0;

        m_data.m_ref_pixel_was_good = 0;

        m_num_of_quick_data = 0;
        m_quick_pos_data = 0;
        m_quick_color_and_var_data = 0;

        m_mean_idepth = 1;
        m_num_points = 0;

        m_num_frames_tracked_on_this = 0;
        m_num_mapped_on_this = 0;
        m_num_mapped_on_this_total = 0;

        m_index_in_keyframes = -1;

        m_edge_error_sum = 1;
        m_edges_num = 1;

        m_last_constraint_tracked_cam_to_world = Sim3();

        m_is_active  = false;
    }

    void c_Frame::build_image(int level)
    {
        if(level == 0)
            return;
        
        build_image(level-1);

        std::unique_lock<std::mutex> build_lock(m_build_mutex);

        if(m_data.m_image_valid[level])
            return;

        int source_width = m_data.m_width[level - 1];
        int source_height = m_data.m_height[level - 1];
        const float* source = m_data.m_image[level - 1];

        if(m_data.m_image[level] == 0)
            m_data.m_image[level] = (float*) c_FrameMemory::get_instance().dispatch_buffer(sizeof(float) * m_data.m_width[level] * m_data.m_height[level]);
        
        float* target = m_data.m_image[level];

        for(int y = 0;y < source_height;y += 2)
            for(int x = 0;x < source_width;x += 2)
            {
                const float* tmp_image = source + x + y*source_width;
                *target = (tmp_image[0] + tmp_image[1] + tmp_image[source_width] + tmp_image[1+source_width]) * 0.25f;
                target++;

            }
        
        m_data.m_image_valid[level] = true;
    }

    void c_Frame::release_image(int level,bool keep_min_level_data,bool only_clean_flag)
    {
        if(level == 0 && keep_min_level_data)
            return;
        


        if(m_data.m_image_valid[level])
        {
            m_data.m_image_valid[level] = false;

            if(!only_clean_flag)
            {        
		if(level == 0) //special for image
            	    return;
                c_FrameMemory::get_instance().reclaim_buffer(m_data.m_image[level]);
                m_data.m_image[level] = 0;
            }
        }

        if(level != 0)
            release_image(level-1,keep_min_level_data,only_clean_flag);
    }

    void c_Frame::build_gradients(int level)
    {
        build_image(level);

        std::unique_lock<std::mutex> build_lock(m_build_mutex);

        if(m_data.m_gradients_valid[level])
            return;
        
        int width = m_data.m_width[level];
        int height = m_data.m_height[level];
        if(m_data.m_gradients[level] == 0)
            m_data.m_gradients[level] = (Eigen::Vector4f*) c_FrameMemory::get_instance().dispatch_buffer(sizeof(Eigen::Vector4f) * width * height);
        Eigen::Vector4f* gradient_it = m_data.m_gradients[level];
        float* image_it = m_data.m_image[level];


        for(int y = 1;y < height-1;y++)
            for(int x = 0;x < width;x++)
            {
                int index = x + y * width;
                float I_r = *(image_it+index+1);
                float I_l = *(image_it+index-1);
                float I_o = *(image_it+index);
                float I_u = *(image_it+index-width);
                float I_d = *(image_it+index+width);
                (*(gradient_it+index))[0] = 0.5f * (I_r - I_l);
                (*(gradient_it+index))[1] = 0.5f * (I_d - I_u);
                (*(gradient_it+index))[2] = I_o;   
            }

        m_data.m_gradients_valid[level] = true;
    }

    void c_Frame::release_gradients(int level,bool keep_min_level_data,bool only_clean_flag)
    {
        if(level == 0 && keep_min_level_data)
            return;

        if(m_data.m_gradients_valid[level])
        {
            m_data.m_gradients_valid[level] = false;

            if(!only_clean_flag)
            {
                c_FrameMemory::get_instance().reclaim_buffer(reinterpret_cast<float*>(m_data.m_gradients[level]));
                m_data.m_gradients[level] = 0;
            }
        }

        if(level != 0)
            release_gradients(level-1,keep_min_level_data,only_clean_flag);
    }

    void c_Frame::build_max_gradients(int level)
    {
        build_gradients(level);

        std::unique_lock<std::mutex> build_lock(m_build_mutex);

        if(m_data.m_max_gradients_valid[level])
            return;

        int width = m_data.m_width[level];
        int height = m_data.m_height[level];
        if(m_data.m_max_gradients[level] == 0)
            m_data.m_max_gradients[level] = (float*) c_FrameMemory::get_instance().dispatch_buffer(sizeof(float)*width*height);
        
        Eigen::Vector4f* gradient_it = m_data.m_gradients[level];
        float* max_gradient_it = m_data.m_max_gradients[level];

        float* tmp_buffer = (float*) c_FrameMemory::get_instance().dispatch_buffer(sizeof(float) * width * height);

        float* tmp_buffer_it = tmp_buffer;

        for(int y = 1;y < height-1;y++)
            for(int x = 0;x < width;x++)
            {
                int index = x + y * width;
                float dx = (*(gradient_it+index))[0];
                float dy = (*(gradient_it+index))[1];
                *(max_gradient_it+index) = sqrtf(dx*dx+dy*dy);
            }

        for(int y = 1;y < height-1;y++)
            for(int x = 0;x < width;x++)
            {
                if(x == 0 &&  y == 1)
                    continue;
                if(x == width - 1 && y == height -1)
                    continue;
                int index = x + y * width;
                float L_u = *(max_gradient_it+index - width);
                float L_o = *(max_gradient_it+index);
                float L_d = *(max_gradient_it+index + width);
                *(tmp_buffer_it+index) = std::max(L_u,std::max(L_o,L_d));
            }
        
        float num_mappable_pixels = 0;

        for(int y = 1;y < height-1;y++)
            for(int x = 0;x < width;x++)
            {
                if(x == 0 &&  y == 1)
                    continue;
                if(x == width - 1 && y == height -1)
                    continue;
                int index = x+y*width;
                float L_l = *(tmp_buffer_it+index-1);
                float L_o = *(tmp_buffer_it+index);
                float L_r = *(tmp_buffer_it+index+1);
                *(max_gradient_it+index) = std::max(L_l,std::max(L_o,L_r));

                if(*(max_gradient_it+index) >= MIN_ABS_GRAD_CREATE )
                    num_mappable_pixels++;
            }
        
        if(level == 0)
            m_num_mappable_pixels = num_mappable_pixels;
        
        c_FrameMemory::get_instance().reclaim_buffer(tmp_buffer);

        m_data.m_max_gradients_valid[level] = true;
    }

    void c_Frame::release_max_gradients(int level,bool keep_min_level_data,bool only_clean_flag)
    {
        if(level == 0 && keep_min_level_data)
            return;

        if(m_data.m_max_gradients_valid[level])
        {
            m_data.m_max_gradients_valid[level] = false;

            if(!only_clean_flag)
            {
                c_FrameMemory::get_instance().reclaim_buffer(m_data.m_max_gradients[level]);
                m_data.m_max_gradients[level] = 0;
            }
        }

        if(level != 0)
            release_max_gradients(level-1,keep_min_level_data,only_clean_flag);
    }

    void c_Frame::build_idepth_and_var(int level)
    {
        if(!m_data.m_has_idepth_been_set)
            return;

        if(level == 0)
            return;
        
        build_idepth_and_var(level-1);

        std::unique_lock<std::mutex> build_lock(m_build_mutex);

        if(m_data.m_idepth_valid[level] && m_data.m_idepth_var_valid[level])
            return;
        
        int width = m_data.m_width[level];
        int height = m_data.m_height[level];

        if(m_data.m_idepth[level] == 0)
            m_data.m_idepth[level] = (float*) c_FrameMemory::get_instance().dispatch_buffer(sizeof(float) * width * height);
        if(m_data.m_idepth_var[level] == 0)
            m_data.m_idepth_var[level] = (float*) c_FrameMemory::get_instance().dispatch_buffer(sizeof(float) * width * height);

        int source_width = m_data.m_width[level-1];

        const float* source_idepth_it = m_data.m_idepth[level-1];
        const float* source_idepth_var_it = m_data.m_idepth_var[level-1];
        float* target_idepth_it = m_data.m_idepth[level];
        float* target_idepth_var = m_data.m_idepth_var[level];

        for(int y = 0;y < height;y++)
            for(int x = 0;x < width;x++)
            {
                int source_index = 2*(x+y*source_width);
                int target_index = x + y * width;

                float sum_idepth = 0;
                float sum_ivar = 0;
                int num = 0;

                int pos[] = {0,1,source_width,source_width+1};//Var_o,Var_r,Var_d,Var_rd
                
                for(int i = 0;i < 4;i++)
                {
                    float tmp_var = *(source_idepth_var_it+source_index+pos[i]);

                    if(tmp_var > 0)
                    {
                        float ivar = 1.0f / tmp_var;
                        sum_ivar += ivar;
                        sum_idepth += ivar * (*(source_idepth_it+source_index+pos[i]));
                        num++;
                    }
                }

                if(num > 0)
                {
                    float depth = sum_ivar / sum_idepth;
                    *(target_idepth_it+target_index) = 1.0f / depth;
                    *(target_idepth_var+target_index) = num / sum_ivar;
                }
		else
		{
		    *(target_idepth_it+target_index) = -1;
                    *(target_idepth_var+target_index) = -1;			
		}
            }
        
        m_data.m_idepth_valid[level] = true;
        m_data.m_idepth_var_valid[level] = true;
    }

    void c_Frame::release_idepth(int level,bool keep_min_level_data,bool only_clean_flag)
    {
        if(level == 0 && keep_min_level_data)
            return;



        if(m_data.m_idepth_valid[level])
        {
            m_data.m_idepth_valid[level] = false;

            if(!only_clean_flag)
            {        
		if(level == 0) //special for idepth
            	    return;
                c_FrameMemory::get_instance().reclaim_buffer(m_data.m_idepth[level]);
                m_data.m_idepth[level] = 0;
            }
        }

        if(level != 0)
            release_idepth(level-1,keep_min_level_data,only_clean_flag);
    }

    void c_Frame::release_idepth_var(int level,bool keep_min_level_data,bool only_clean_flag)
    {
        if(level == 0 && keep_min_level_data)
            return;



        if(m_data.m_idepth_var_valid[level])
        {
            m_data.m_idepth_var_valid[level] = false;


            if(!only_clean_flag)
            {
		if(level == 0) //special for idepth var
            	    return;
                c_FrameMemory::get_instance().reclaim_buffer(m_data.m_idepth_var[level]);
                m_data.m_idepth_var[level] = 0;
            }
        }

        if(level != 0)
            release_idepth_var(level-1,keep_min_level_data,only_clean_flag);
    }

    bool c_Frame::minimize_memory()
    {
        if(m_active_mutex.try_write_lock_for(10))
        {
            m_build_mutex.lock();

            release_image(PYRAMID_LEVELS-1,true,false);
            release_idepth(PYRAMID_LEVELS-1,true,false);
            release_idepth_var(PYRAMID_LEVELS-1,true,false);

            release_gradients(PYRAMID_LEVELS-1,false,false);
            release_max_gradients(PYRAMID_LEVELS-1,false,false);

            clear_ref_pixel_was_good();

            m_build_mutex.unlock();
            m_active_mutex.unlock_try_write_lock();
            return true;
        }
        return false;
    }
}
