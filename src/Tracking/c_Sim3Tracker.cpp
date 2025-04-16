#include "c_Sim3Tracker.h"
#include "DataStructures/c_Frame.h"
#include "Tracking/c_TrackingReference.h"
#include "util/interpolation.h"


namespace lsd_slam
{
    c_Sim3Tracker::c_Sim3Tracker(int w,int h,Eigen::Matrix3f K)
    {
        m_width = w;
        m_height = h;

        this->m_K = K;
        m_fx = K(0,0);m_fy = K(1,1);m_cx = K(0,2);m_cy = K(1,2);

        m_hyperparameters = DenseDepthTrackerSettings();

        m_KInv = K.inverse();
        m_fxi = m_KInv(0,0);m_fyi = m_KInv(1,1);m_cxi = m_KInv(0,2);m_cyi = m_KInv(1,2);

        m_frame_point_pos_x = (float*)Eigen::internal::aligned_malloc(w*h*sizeof(float));
        m_frame_point_pos_y = (float*)Eigen::internal::aligned_malloc(w*h*sizeof(float));
        m_frame_point_pos_z = (float*)Eigen::internal::aligned_malloc(w*h*sizeof(float));
        m_frame_gradient_dx = (float*)Eigen::internal::aligned_malloc(w*h*sizeof(float));
        m_frame_gradient_dy = (float*)Eigen::internal::aligned_malloc(w*h*sizeof(float));

        m_residual_depth = (float*)Eigen::internal::aligned_malloc(w*h*sizeof(float));
        m_residual_photometric = (float*)Eigen::internal::aligned_malloc(w*h*sizeof(float));
        
        m_weighted_depth = (float*)Eigen::internal::aligned_malloc(w*h*sizeof(float));
        m_weighted_photometric = (float*)Eigen::internal::aligned_malloc(w*h*sizeof(float));
        

        m_num_index = 0;

        m_point_usage = 0;

        m_last_residual = 0;m_last_depth_residual = 0;m_last_photometric_residual = 0;
        m_last_residual_unweighted = 0;m_last_depth_residual_unweighted = 0;m_last_photometric_residual_unweighted = 0;
    }

    c_Sim3Tracker::~c_Sim3Tracker()
    {

        Eigen::internal::aligned_free((void*)m_frame_point_pos_x);
        Eigen::internal::aligned_free((void*)m_frame_point_pos_y);
        Eigen::internal::aligned_free((void*)m_frame_point_pos_z);
        Eigen::internal::aligned_free((void*)m_frame_gradient_dx);
        Eigen::internal::aligned_free((void*)m_frame_gradient_dy);

        Eigen::internal::aligned_free((void*)m_residual_depth);
        Eigen::internal::aligned_free((void*)m_residual_photometric);

        Eigen::internal::aligned_free((void*)m_weighted_depth);
        Eigen::internal::aligned_free((void*)m_weighted_photometric);

    }

    Sim3 c_Sim3Tracker::trackFrame(
        c_TrackingReference* reference,c_Frame* frame,const Sim3& frame_to_reference_initial,int start_level,int end_level)
    {
        c_read_write_lock* lock = frame->get_active_lock();

        Sim3 reference_to_frame = frame_to_reference_initial.inverse();

        m_diverged = false;

        m_affine_trans_a = 1;m_affine_trans_b = 0;

        c_sim3_residual final_residual;

        bool update = false;
        Vector7f b;
        Matrix7f A;
        for(int level = start_level;level >= end_level;level--)
        {
            if(m_hyperparameters.maxItsPerLvl[level] == 0)
                continue;
            
            reference->make_pointcloud(level);
            c_sim3_residual error = calculate_error(reference,frame,reference_to_frame,level);

            if(m_num_index < 0.5*MIN_GOODPERALL_PIXEL_ABSMIN * (m_width>>level)*(m_height>>level)||m_num_index<10)
            {
                m_diverged = true;
                lock->unlock_shared();
                return Sim3();
            }

            if(useAffineLightningEstimation)
            {
                m_affine_trans_a = affine_trans_a_last_it;
                m_affine_trans_b = affine_trans_b_last_it;
            }

            float lambda = m_hyperparameters.lambdaInitial[level];
            update = false;
            for(int it = 0;it < m_hyperparameters.maxItsPerLvl[level];it++)
            {
                b.setZero();
                A.setZero();
                Eigen::Matrix3f K_level = frame->get_K(level);
                float fx_level = K_level(0,0);
                float fy_level = K_level(1,1);
                calculate_iterate(A,b,fx_level,fy_level);

		update = true;

                int pow_of_lamda = 0;
                while(true)
                {
                    Matrix7f tmp_A = A;
                    for(int i = 0;i < 7;i++)
                        tmp_A(i,i) *= 1+lambda;
                    Vector7f increase_of_ksai = tmp_A.ldlt().solve(b);
                    pow_of_lamda++;

                    float abs_increase = increase_of_ksai.dot(increase_of_ksai);
                    if(!(abs_increase >= 0 && abs_increase < 1))
                    {
                        m_last_sim3_hessian.setZero();
                        lock->unlock_shared();
                        return Sim3();
                    }

                    Sim3 candidate_reference_to_frame = Sim3::exp(increase_of_ksai.cast<sophusType>())*reference_to_frame;

                    c_sim3_residual tmp_error = calculate_error(reference,frame,candidate_reference_to_frame,level);

                    if(m_num_index < 0.5*MIN_GOODPERALL_PIXEL_ABSMIN * (m_width>>level)*(m_height>>level)||m_num_index<10)
                    {
                        m_diverged = true;
                        lock->unlock_shared();
                        return Sim3();

                    }

                    if(tmp_error.m_mean < error.m_mean)
                    {
                        reference_to_frame = candidate_reference_to_frame;
                        update = false;

                        if(useAffineLightningEstimation)
                        {
                            m_affine_trans_a = affine_trans_a_last_it;
                            m_affine_trans_b = affine_trans_b_last_it;
                        }

                        if(tmp_error.m_mean / error.m_mean > m_hyperparameters.convergenceEps[level])
                        {
                            it = m_hyperparameters.maxItsPerLvl[level];
                        }
                        final_residual = error = tmp_error;

                        if(lambda <= 0.2)
                            lambda = 0;
                        else
                            lambda *= m_hyperparameters.lambdaSuccessFac;
                        break;
                    }
                    else
                    {
                        if(!(abs_increase > m_hyperparameters.stepSizeMin[level]))
                        {
                            it = m_hyperparameters.maxItsPerLvl[level];
                            break;
                        }
                        
                        if(lambda == 0)
                            lambda = 0.2;
                        else
                            lambda *= std::pow(m_hyperparameters.lambdaFailFac,pow_of_lamda);
                    }

                }
            }


        }

        if(!update)
        {
            reference->make_pointcloud(end_level);
            final_residual = calculate_error(reference,frame,reference_to_frame,end_level);
            b.setZero();
            A.setZero();
            Eigen::Matrix3f K_level = frame->get_K(end_level);
            float fx_level = K_level(0,0);
            float fy_level = K_level(1,1);
            calculate_iterate(A,b,fx_level,fy_level);
        }
        //m_last_sim3_hessian = A * 2*m_num_index;

        if(reference_to_frame.scale() <= 0)
        {
            m_diverged = true;
            lock->unlock_shared();
            return Sim3();
        }

        m_last_residual = final_residual.m_mean;
        m_last_depth_residual = final_residual.m_mean_depth;
        m_last_photometric_residual = final_residual.m_mean_photometric;
        

        lock->unlock_shared();
        return reference_to_frame.inverse();
    }

    c_sim3_residual c_Sim3Tracker::calculate_error(c_TrackingReference* reference,c_Frame* frame,Sim3& reference_to_frame,int level)
    {
        int w = frame->get_width(level);
        int h = frame->get_height(level);
        Eigen::Matrix3f K_level = frame->get_K(level);
        float fx_level = K_level(0,0);
        float fy_level = K_level(1,1);
        float cx_level = K_level(0,2);
        float cy_level = K_level(1,2);

        Eigen::Matrix3f rotation_matrix = reference_to_frame.rxso3().matrix().cast<float>();
        Eigen::Matrix3f rotation_matrix_unscaled = reference_to_frame.rotationMatrix().cast<float>();
        Eigen::Vector3f translation = reference_to_frame.translation().cast<float>();

        //calculate rotation around optical axis and axis' rotation, used for ESM
        Eigen::Vector3f forward_vector(0,0,-1);
        Eigen::Vector3f rotation_forward_vector = rotation_matrix_unscaled * forward_vector;
        Eigen::Quaternionf shortest_back_rotation;
        shortest_back_rotation.setFromTwoVectors(rotation_forward_vector,forward_vector);
        Eigen::Matrix3f roll_matrix = shortest_back_rotation.toRotationMatrix() * rotation_matrix_unscaled;
        float x_roll_0 = roll_matrix(0,0);
        float x_roll_1 = roll_matrix(0,1);
        float y_roll_0 = roll_matrix(1,0);
        float y_roll_1 = roll_matrix(1,1);

        int num = reference->m_num_data[level];
        
        Eigen::Vector3f* reference_point_pos = reference->m_pos_data[level];
        Eigen::Vector2f* reference_color_and_var = reference->m_color_and_var_data[level];
        Eigen::Vector2f* reference_gradient = reference->m_grad_data[level];

        const float* frame_idepth = frame->get_idepth(level);
        const float* frame_idepth_var = frame->get_idepth_var(level);
        const Eigen::Vector4f* frame_intensity_and_gradient = frame->get_gradients(level);

        float sxx=0,syy=0,sx=0,sy=0,sw=0;

        float usage_count = 0;

        int index = 0;

        c_sim3_residual sum_residual;
        
        
        for(int i = 0;i < num;i++,reference_point_pos++,reference_color_and_var++,reference_gradient++)
        {
            Eigen::Vector3f cur_point_pos = rotation_matrix * (*reference_point_pos) + translation;
            float u_new = (cur_point_pos[0]/cur_point_pos[2]) * fx_level+cx_level;
            float v_new = (cur_point_pos[1]/cur_point_pos[2]) * fy_level+cy_level;

            if(!(u_new > 1 && v_new > 1 && u_new < w-2 && v_new < h-2))
            {
                continue;
            }

            *(m_frame_point_pos_x+index) = cur_point_pos[0];
            *(m_frame_point_pos_y+index) = cur_point_pos[1];
            *(m_frame_point_pos_z+index) = cur_point_pos[2]; 

            Eigen::Vector4f frame_gradient_and_color = get_interpolated_element<Eigen::Vector4f>(frame_intensity_and_gradient,u_new,v_new,w);

            //ESM
            float rotation_grad_x = x_roll_0 * (*reference_gradient)[0]+x_roll_1 * (*reference_gradient)[1];
            float rotation_grad_y = y_roll_0 * (*reference_gradient)[0]+y_roll_1 * (*reference_gradient)[1];

            *(m_frame_gradient_dx+index) = fx_level * 0.5f * (frame_gradient_and_color[0]+rotation_grad_x);
            *(m_frame_gradient_dy+index) = fy_level * 0.5f * (frame_gradient_and_color[1]+rotation_grad_y);

            //photometric residual
            float I_ref = m_affine_trans_a * (*reference_color_and_var)[0] + m_affine_trans_b;
            float I_frame = frame_gradient_and_color[2];
            float residual_photometric = I_ref - I_frame;
            *(m_residual_photometric+index) = residual_photometric;

            //affine
            float weight = fabsf(residual_photometric) < 2.0f ? 1 : 2.0f / fabsf(residual_photometric);
            sxx += I_ref * I_ref * weight;
            syy += I_frame * I_frame * weight;
            sx += I_ref * weight;
            sy += I_frame * weight;
            sw += weight;

            //depth residual
            int new_index = (int)(u_new+0.5f)+w*(int)(v_new+0.5f);
            float var_frame_depth = frame_idepth_var[new_index];
            float idepth_trans_frame = 1.0f / cur_point_pos[2];
            float residual_depth;
            if(var_frame_depth > 0)
            {
                residual_depth = idepth_trans_frame - frame_idepth[new_index];
                *(m_residual_depth+index) = residual_depth;
            }
            else
            {
                residual_depth = -1;
                *(m_residual_depth+index) = -1;
            }

            //--------------------calculate residual var------------
            float tx = translation[0];
            float ty = translation[1];
            float tz = translation[2];

            float px = cur_point_pos[0];
            float py = cur_point_pos[1];
            float pz = cur_point_pos[2];

            float d = 1.0f / (*reference_point_pos)[2];

            float rp = residual_photometric;
            float rd = residual_depth;

            float gx = *(m_frame_gradient_dx+index);
            float gy = *(m_frame_gradient_dy+index);

            float var_depth_ref = m_hyperparameters.var_weight * (*reference_color_and_var)[1];
            float var_depth_frame = m_hyperparameters.var_weight * var_frame_depth;

            float partial_rp_div_partial_d = (gx * ((tx * pz - tz * px) / (pz*pz*d))+gy*((ty*pz-tz*py)/(pz*pz*d)));
            float var_rp = ((cameraPixelNoise2)+ var_depth_ref*partial_rp_div_partial_d * partial_rp_div_partial_d);
            float weight_rp = 1.0f/var_rp;

            float partial_rd_div_partial_d = (pz-tz)/(pz*pz*d);
            float var_rd = var_depth_frame + var_depth_ref * partial_rd_div_partial_d * partial_rd_div_partial_d;
            //float weight_rd = 1.0f/var_rd;
            float weight_rd = 1.0f/(var_depth_frame+partial_rd_div_partial_d * partial_rd_div_partial_d * var_depth_ref);

            float weighted_rp = fabs(rp*sqrtf(weight_rp));
            float weighted_rd = fabs(rd*sqrtf(weight_rd));

            float weighted_abs_res = var_depth_frame > 0 ? weighted_rd+weighted_rp : weighted_rp;
            
            float weight_huber = fabs(weighted_abs_res < m_hyperparameters.huber_d ? 1 : m_hyperparameters.huber_d / weighted_abs_res);



            *(m_weighted_photometric+index) = weight_huber * weight_rp;
            if(var_depth_frame > 0)
            {
                *(m_weighted_depth+index) = weight_huber * weight_rd;
            }
            else
            {
                *(m_weighted_depth+index) = 0;
            }

            if(var_depth_frame > 0)
            {
                sum_residual.m_sum_res_depth += weight_huber * weight_rd * rd * rd;
                sum_residual.m_num_depth++;
            }

            sum_residual.m_sum_res_photometric += weight_huber * weight_rp * rp * rp;
            sum_residual.m_num_photometric++;

            float depth_change = (*reference_point_pos)[2] / cur_point_pos[2];
            usage_count += depth_change < 1 ? depth_change : 1;

            index++;
        }
        m_num_index = index;

        m_point_usage = usage_count / (float)num;

        affine_trans_a_last_it = sqrtf((syy - sy*sy/sw) / (sxx - sx*sx/sw));
        affine_trans_b_last_it = (sy - affine_trans_a_last_it * sx) / sw;

        sum_residual.m_mean = (sum_residual.m_sum_res_depth+sum_residual.m_sum_res_photometric) / (sum_residual.m_num_depth+sum_residual.m_num_photometric);
        sum_residual.m_mean_depth = (sum_residual.m_sum_res_depth) / (sum_residual.m_num_depth);
        sum_residual.m_mean_photometric = (sum_residual.m_sum_res_photometric) / (sum_residual.m_num_photometric);

        return sum_residual;
    }

    void c_Sim3Tracker::calculate_iterate(Matrix7f& A,Vector7f& b,float fx_level,float fy_level)
    {
        Matrix7f A_d;A_d.setZero();
        Vector7f b_d;b_d.setZero();
        for(int i = 0;i < m_num_index;i++)
        {
            float px = *(m_frame_point_pos_x+i);
            float py = *(m_frame_point_pos_y+i);
            float pz = *(m_frame_point_pos_z+i);

            float wp = *(m_weighted_photometric+i);
            float wd = *(m_weighted_depth+i);
            
            float rp = *(m_residual_photometric+i);
            float rd = *(m_residual_depth+i);

            float gx = *(m_frame_gradient_dx+i);
            float gy = *(m_frame_gradient_dy+i);

            float z = 1.0f/pz;
            float z_sqr = 1.0f/(pz*pz);

            Vector7f J_p;
            Vector7f J_d;

            J_p[0] = gx * z                       + 0                              ;
            J_p[1] = 0                            + gy * z                         ;
            J_p[2] = gx * (-px * z_sqr)           + gy * (-py * z_sqr)             ;
            J_p[3] = gx * (-px * py * z_sqr)      + gy * (-(1.0 + py * py * z_sqr));
            J_p[4] = gx * (1.0 + px * px * z_sqr) + gy * (px * py * z_sqr)         ;
            J_p[5] = gx * (-py * z)               + gy * (px * z)                  ;
            J_p[6] = 0;

            J_d[0] = 0;
            J_d[1] = 0;
            J_d[2] = z_sqr;
            J_d[3] = z_sqr * py;
            J_d[4] = -z_sqr * px;
            J_d[5] = 0;
            J_d[6] = z;

            A.noalias() += J_p * J_p.transpose() * wp;
            b.noalias() -= J_p * (rp * wp);

            //A.noalias() += J_d * J_d.transpose() * wd;
            //b.noalias() -= J_d * (rd * wd);
            A_d.noalias() += J_d * J_d.transpose() * wd;
            b_d.noalias() -= J_d * (rd * wd);
        }
        A += A_d;
        b += b_d;
	
	m_last_sim3_hessian = A;	
	
        A = A / (2*m_num_index);
        b = -b/ (2*m_num_index);
    }

}
