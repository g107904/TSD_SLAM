#include "c_SE3Tracker.h"
#include "DataStructures/c_Frame.h"
#include "Tracking/c_TrackingReference.h"
#include "util/interpolation.h"

#include <omp.h>
#include <Eigen/Core>



namespace lsd_slam
{
    c_SE3Tracker::c_SE3Tracker(int w, int h, Eigen::Matrix3f K)
    {
        m_width = w;
        m_height = h;

        this->m_K = K;

        m_hyperparameters = DenseDepthTrackerSettings();



        m_frame_point_pos_x = (float*)Eigen::internal::aligned_malloc(w * h * sizeof(float));
        m_frame_point_pos_y = (float*)Eigen::internal::aligned_malloc(w * h * sizeof(float));
        m_frame_point_pos_z = (float*)Eigen::internal::aligned_malloc(w * h * sizeof(float));
        m_frame_gradient_dx = (float*)Eigen::internal::aligned_malloc(w * h * sizeof(float));
        m_frame_gradient_dy = (float*)Eigen::internal::aligned_malloc(w * h * sizeof(float));
        m_frame_idepth = (float*)Eigen::internal::aligned_malloc(w * h * sizeof(float));
        m_frame_idepth_var = (float*)Eigen::internal::aligned_malloc(w * h * sizeof(float));
        m_frame_residual = (float*)Eigen::internal::aligned_malloc(w * h * sizeof(float));
        m_frame_residual_var = (float*)Eigen::internal::aligned_malloc(w * h * sizeof(float));
        m_num_index = 0;

        m_point_usage = 0;

        m_last_residual = m_last_mean_residual = 0;
        m_last_good_count = m_last_bad_count = 0;

        m_diverged = false;

        m_flag = 0;
    }
    c_SE3Tracker::~c_SE3Tracker()
    {
        Eigen::internal::aligned_free((void*)m_frame_point_pos_x);
        Eigen::internal::aligned_free((void*)m_frame_point_pos_y);
        Eigen::internal::aligned_free((void*)m_frame_point_pos_z);
        Eigen::internal::aligned_free((void*)m_frame_gradient_dx);
        Eigen::internal::aligned_free((void*)m_frame_gradient_dy);
        Eigen::internal::aligned_free((void*)m_frame_idepth);
        Eigen::internal::aligned_free((void*)m_frame_idepth_var);
        Eigen::internal::aligned_free((void*)m_frame_residual);
        Eigen::internal::aligned_free((void*)m_frame_residual_var);
    }

    float c_SE3Tracker::check_overlap(c_Frame* reference, SE3& reference_to_frame)
    {
        Sophus::SE3f reference_to_frame_float = reference_to_frame.cast<float>();
        std::unique_lock<std::mutex> lock2 = std::unique_lock<std::mutex>(reference->m_quick_mutex);

        int w2 = reference->get_width(QUICK_KF_CHECK_LVL) - 1;
        int h2 = reference->get_height(QUICK_KF_CHECK_LVL) - 1;
        Eigen::Matrix3f K_level = reference->get_K(QUICK_KF_CHECK_LVL);
        float fx_level = K_level(0, 0);
        float fy_level = K_level(1, 1);
        float cx_level = K_level(0, 2);
        float cy_level = K_level(1, 2);

        Eigen::Matrix3f rotation_matrix = reference_to_frame_float.rotationMatrix();
        Eigen::Vector3f translation = reference_to_frame_float.translation();

        int num = reference->m_num_of_quick_data;
        const Eigen::Vector3f* reference_point_pos = reference->m_quick_pos_data;

        float usage_count = 0;
        for (int i = 0; i < num; reference_point_pos++, i++)
        {
            Eigen::Vector3f cur_point_pos = rotation_matrix * (*reference_point_pos) + translation;

            float u_new = (cur_point_pos[0] / cur_point_pos[2]) * fx_level + cx_level;
            float v_new = (cur_point_pos[1] / cur_point_pos[2]) * fy_level + cy_level;

            if ((u_new > 0 && v_new > 0 && u_new < w2 && v_new < h2))
            {
                float depth_change = (*reference_point_pos)[2] / cur_point_pos[2];
                usage_count += depth_change < 1 ? depth_change : 1;
            }
        }

        m_point_usage = usage_count / (float)num;
        return m_point_usage;
    }

    SE3 c_SE3Tracker::quick_trackFrame(c_Frame* reference, c_Frame* frame, SE3& reference_to_frame_init)
    {
        Sophus::SE3f reference_to_frame = reference_to_frame_init.cast<float>();


        c_read_write_lock* lock = frame->get_active_lock();
        std::unique_lock<std::mutex> lock2 = std::unique_lock<std::mutex>(reference->m_quick_mutex);

        m_diverged = false;
        m_tracking_was_good = true;
        m_affine_estimate_a = 1; m_affine_estimate_b = 0;

        float error = quick_calculate_error(reference, frame, reference_to_frame);

        if (m_num_index < MIN_GOODPERALL_PIXEL_ABSMIN * (m_width >> QUICK_KF_CHECK_LVL) * (m_height >> QUICK_KF_CHECK_LVL))
        {
            m_diverged = true;
            m_tracking_was_good = false;

            lock->unlock_shared();

            return SE3();
        }

        if (useAffineLightningEstimation)
        {
            m_affine_estimate_a = m_affine_estimate_a_last_it;
            m_affine_estimate_b = m_affine_estimate_b_last_it;
        }

        float tmp_last_residual = 0;

        float lambda = m_hyperparameters.lambdaInitialTestTrack;
        int level = QUICK_KF_CHECK_LVL;
        for (int it = 0; it < m_hyperparameters.maxItsTestTrack; it++)
        {
            Vector6f b;
            Matrix6f A;
            Eigen::Matrix3f K_level = m_K;
            float fx_level = K_level(0, 0);
            float fy_level = K_level(1, 1);
            calculate_iterate(A, b, fx_level, fy_level);

            int pow_of_lambda = 0;
            while (true)
            {
                Matrix6f tmp_A = A;
                for (int i = 0; i < 6; i++)
                    tmp_A(i, i) *= (1 + lambda);
                Vector6f increase_of_ksai = tmp_A.ldlt().solve(b);
                pow_of_lambda++;

                Sophus::SE3f candidate_reference_to_frame = Sophus::SE3f::exp((increase_of_ksai)) * reference_to_frame;
                float tmp_error = quick_calculate_error(reference, frame, candidate_reference_to_frame);

                if (m_num_index < MIN_GOODPERALL_PIXEL_ABSMIN * (m_width >> level) * (m_height >> level))
                {
                    m_diverged = true;
                    m_tracking_was_good = false;

                    lock->unlock_shared();

                    return SE3();
                }
                if (tmp_error < error)
                {
                    reference_to_frame = candidate_reference_to_frame;

                    if (useAffineLightningEstimation)
                    {
                        m_affine_estimate_a = m_affine_estimate_a_last_it;
                        m_affine_estimate_b = m_affine_estimate_b_last_it;
                    }

                    if (tmp_error / error > m_hyperparameters.convergenceEpsTestTrack)
                    {
                        it = m_hyperparameters.maxItsTestTrack;
                    }
                    tmp_last_residual = error = tmp_error;

                    if (lambda <= 0.2)
                        lambda = 0;
                    else
                        lambda *= m_hyperparameters.lambdaSuccessFac;
                    break;
                }
                else
                {

                    if (!(increase_of_ksai.dot(increase_of_ksai) > m_hyperparameters.stepSizeMinTestTrack))
                    {


                        it = m_hyperparameters.maxItsTestTrack;
                        break;
                    }
                    if (lambda == 0)
                        lambda = 0.2;
                    else
                        lambda *= std::pow(m_hyperparameters.lambdaFailFac, pow_of_lambda);
                }
            }
        }


        m_last_residual = tmp_last_residual;

        float good_in_all = m_last_good_count / (frame->get_width(level) * frame->get_height(level));
        float good_in_count = m_last_good_count / (m_last_good_count + m_last_bad_count);
        m_tracking_was_good = !m_diverged && (good_in_all > MIN_GOODPERALL_PIXEL) && (good_in_count > MIN_GOODPERGOODBAD_PIXEL);

        lock->unlock_shared();

        return toSophus(reference_to_frame);


    }

    float c_SE3Tracker::quick_calculate_error(c_Frame* reference, c_Frame* frame, const Sophus::SE3f& reference_to_frame)
    {
        int level = QUICK_KF_CHECK_LVL;

        Eigen::Vector3f* reference_point_pos = reference->m_quick_pos_data;
        Eigen::Vector2f* reference_color_and_var = reference->m_quick_color_and_var_data;
        int num = reference->m_num_of_quick_data;
        int w = frame->get_width(level);
        int h = frame->get_height(level);
        Eigen::Matrix3f K_level = frame->get_K(level);
        float fx_level = K_level(0, 0);
        float fy_level = K_level(1, 1);
        float cx_level = K_level(0, 2);
        float cy_level = K_level(1, 2);
        Eigen::Matrix3f rotation = reference_to_frame.rotationMatrix();
        Eigen::Vector3f trans = reference_to_frame.translation();
        const Eigen::Vector4f* frame_gradients = frame->get_gradients(level);

        int index = 0;
        float tmp_sum_error = 0;

        int good_count = 0, bad_count = 0;
        float sum_signed_res = 0;

        float sxx = 0, syy = 0, sx = 0, sy = 0, sw = 0;
        float usage_count = 0;

        for (int i = 0; i < num; reference_point_pos++, reference_color_and_var++, i++)
        {
            Eigen::Vector3f cur_point_pos = rotation * (*reference_point_pos) + trans;
            float u_new = (cur_point_pos[0] / cur_point_pos[2]) * fx_level + cx_level;
            float v_new = (cur_point_pos[1] / cur_point_pos[2]) * fy_level + cy_level;

            if (!(u_new > 1 && v_new > 1 && u_new < w - 2 && v_new < h - 2))
            {
                continue;
            }

            Eigen::Vector4f frame_gradient_and_color = get_interpolated_element<Eigen::Vector4f>(frame_gradients, u_new, v_new, w);

            float residual_reference = m_affine_estimate_a * (*reference_color_and_var)[0] + m_affine_estimate_b;
            float residual_frame = frame_gradient_and_color[2];
            float residual = residual_reference - residual_frame;

            *(m_frame_point_pos_x + index) = cur_point_pos[0];
            *(m_frame_point_pos_y + index) = cur_point_pos[1];
            *(m_frame_point_pos_z + index) = cur_point_pos[2];
            *(m_frame_gradient_dx + index) = fx_level * frame_gradient_and_color[0];
            *(m_frame_gradient_dy + index) = fy_level * frame_gradient_and_color[1];
            *(m_frame_idepth + index) = 1.0f / (*reference_point_pos)[2];
            *(m_frame_idepth_var + index) = (*reference_color_and_var)[1];
            *(m_frame_residual + index) = residual;

            float weight = fabsf(residual) < 5.0f ? 1 : 5.0f / fabsf(residual);
            sxx += residual_reference * residual_reference * weight;
            syy += residual_frame * residual_frame * weight;
            sx += residual_reference * weight;
            sy += residual_frame * weight;
            sw += weight;

            float residual_threshold =
                MAX_DIFF_CONSTANT
                + MAX_DIFF_GRAD_MULT
                * (frame_gradient_and_color[0] * frame_gradient_and_color[0]
                    + frame_gradient_and_color[1] * frame_gradient_and_color[1]);

            bool is_good = residual * residual / residual_threshold < 1;

            float cur_x_hat = cur_point_pos[0];
            float cur_y_hat = cur_point_pos[1];
            float cur_z_hat = cur_point_pos[2];
            float cur_d = 1.0f / (*reference_point_pos)[2];
            float cur_rp = residual;
            float cur_dx = frame_gradient_and_color[0];
            float cur_dy = frame_gradient_and_color[1];
            float cur_gx = *(m_frame_gradient_dx + index);
            float cur_gy = *(m_frame_gradient_dy + index);
            float cur_sigma_d = (*reference_color_and_var)[1];
            float cur_tx = trans[0];
            float cur_ty = trans[1];
            float cur_tz = trans[2];

            float cur_patial_rp_div_patial_d =
                cur_gx * ((cur_tx * cur_z_hat - cur_tz * cur_x_hat) / (cur_z_hat * cur_z_hat * cur_d))
                + cur_gy * ((cur_ty * cur_z_hat - cur_tz * cur_y_hat) / (cur_z_hat * cur_z_hat * cur_d));

            float sigma_rp = ((cameraPixelNoise2)+cur_sigma_d * cur_patial_rp_div_patial_d * cur_patial_rp_div_patial_d);
            /*
                float r_abs = fabs(cur_rp/sqrt(sigma_rp));
                float tmp_error;
                if(r_abs < (m_hyperparameters.huber_d/2))
                {
                    tmp_error = cur_rp*cur_rp / sigma_rp;
                }
                else
                {
                    tmp_error = fabs(m_hyperparameters.huber_d/2/r_abs)*cur_rp*cur_rp/sigma_rp;
                }
            */
            float w_p = 1.0f / sigma_rp;

            float weighted_rp = fabs(cur_rp * sqrtf(w_p));

            float wh = fabs(weighted_rp < (m_hyperparameters.huber_d / 2) ? 1 : (m_hyperparameters.huber_d / 2) / weighted_rp);

            float tmp_error = wh * w_p * cur_rp * cur_rp;

            tmp_sum_error += tmp_error;
            //*(m_frame_residual_var+index) = sigma_rp;
            *(m_frame_residual_var + index) = wh * w_p;
            index++;

            if (is_good)
            {
                sum_signed_res += residual;
                good_count++;
            }
            else
                bad_count++;

            float depth_change = (*reference_point_pos)[2] / cur_point_pos[2];
            usage_count += depth_change < 1 ? depth_change : 1;
        }
        m_num_index = index;

        m_point_usage = usage_count / num;
        m_last_good_count = good_count;
        m_last_bad_count = bad_count;
        m_last_mean_residual = sum_signed_res / good_count;

        m_affine_estimate_a_last_it = sqrtf((syy - sy * sy / sw) / (sxx - sx * sx * sw));
        m_affine_estimate_b_last_it = (sy - m_affine_estimate_a_last_it * sw) / sw;

        return tmp_sum_error / (float)m_num_index;
    }

    SE3 c_SE3Tracker::trackFrame(c_TrackingReference* reference, c_Frame* frame, const SE3& initial)
    {
        c_read_write_lock* lock = frame->get_active_lock();
        Sophus::SE3f reference_to_frame = initial.inverse().cast<float>();

        m_diverged = false;
        m_tracking_was_good = true;
        m_affine_estimate_a = 1; m_affine_estimate_b = 0;

        float tmp_last_residual = 0;



        for (int level = SE3TRACKING_MAX_LEVEL - 1; level >= SE3TRACKING_MIN_LEVEL; level--)
        {
            reference->make_pointcloud(level);
            float error = calculate_error(reference, frame, reference_to_frame, level);
            //std::cout<<"my:"<<error<<std::endl;
            float lambda = m_hyperparameters.lambdaInitial[level];
            if (m_num_index < MIN_GOODPERALL_PIXEL_ABSMIN * (m_width >> level) * (m_height >> level))
            {
                m_diverged = true;
                m_tracking_was_good = false;

                lock->unlock_shared();
                return SE3();
            }

            if (useAffineLightningEstimation)
            {
                m_affine_estimate_a = m_affine_estimate_a_last_it;
                m_affine_estimate_b = m_affine_estimate_b_last_it;
            }

            for (int it = 0; it < m_hyperparameters.maxItsPerLvl[level]; it++)
            {
                Vector6f b;
                Matrix6f A;
                Eigen::Matrix3f K_level = frame->get_K(level);
                float fx_level = K_level(0, 0);
                float fy_level = K_level(1, 1);
                calculate_iterate(A, b, fx_level, fy_level);

                int pow_of_lambda = 0;
                while (true)
                {
                    Matrix6f tmp_A = A;
                    Vector6f increase_of_ksai;
                    for (int i = 0; i < 6; i++)
                        tmp_A(i, i) *= (1 + lambda);
                    if (m_flag)
                    {
                        if (m_flag < 3)
                        {
                            Eigen::Matrix<float, 4, 4> tmp_mid_A;
                            Eigen::Matrix<float, 4, 1> tmp_mid_b;
                            int flag_pos = m_flag - 1;
                            if (flag_pos == 0)
                            {
                                tmp_mid_b[0] = b[1];
                                tmp_mid_A(0, 0) = tmp_A(1, 1);
                                for (int i = 1; i < 4; i++)
                                {
                                    tmp_mid_A(0, i) = tmp_A(1, i + 2);
                                    tmp_mid_A(i, 0) = tmp_A(i + 2, 1);
                                }

                            }
                            else
                            {
                                tmp_mid_b[0] = b[0];
                                tmp_mid_A(0, 0) = tmp_A(0, 0);
                                for (int i = 1; i < 4; i++)
                                {
                                    tmp_mid_A(0, i) = tmp_A(0, i + 2);
                                    tmp_mid_A(i, 0) = tmp_A(i + 2, 0);
                                }
                            }
                            for (int i = 1; i < 4; i++)
                                for (int j = 1; j < 4; j++)
                                    tmp_mid_A(i, j) = tmp_A(i + 2, j + 2);
                            for (int i = 1; i < 4; i++)
                                tmp_mid_b[i] = b[i + 2];
                            Eigen::Matrix<float, 4, 1> tmp_increase = tmp_mid_A.ldlt().solve(tmp_mid_b);
                            for (int i = 3; i < 6; i++)
                                increase_of_ksai[i] = tmp_increase[i - 2];
                            if (flag_pos == 0)
                                increase_of_ksai[1] = tmp_increase[0];
                            else
                                increase_of_ksai[0] = tmp_increase[0];
                            increase_of_ksai[flag_pos] = 0;
                            increase_of_ksai[2] = 0;
                        }
                        else if (m_flag == 3)
                        {
                            Eigen::Matrix<float, 5, 5> tmp_mid_A;
                            Eigen::Matrix<float, 5, 1> tmp_mid_b;
                            for (int i = 0; i < 2; i++)
                                for (int j = 0; j < 2; j++)
                                    tmp_mid_A(i, j) = tmp_A(i, j);
                            for (int i = 2; i < 5; i++)
                                for (int j = 2; j < 5; j++)
                                    tmp_mid_A(i, j) = tmp_A(i + 1, j + 1);
                            for (int i = 0; i < 2; i++)
                            {
                                for (int j = 2; j < 5; j++)
                                {
                                    tmp_mid_A(i, j) = tmp_A(i, j + 1);
                                    tmp_mid_A(j, i) = tmp_A(j + 1, i);
                                }
                            }
                            for (int i = 0; i < 2; i++)
                                tmp_mid_b[i] = b[i];
                            for (int i = 2; i < 5; i++)
                                tmp_mid_b[i] = b[i + 1];
                            Eigen::Matrix<float, 5, 1> tmp_inc = tmp_mid_A.ldlt().solve(tmp_mid_b);
                            for (int i = 0; i < 2; i++)
                                increase_of_ksai[i] = tmp_inc[i];
                            for (int i = 3; i < 6; i++)
                                increase_of_ksai[i] = tmp_inc[i - 1];
                            increase_of_ksai[2] = 0;
                        }
                        else if (m_flag == 4)
                        {
                            Eigen::Matrix<float, 3, 3> tmp_mid_A;
                            Eigen::Vector3f tmp_mid_b;
                            for (int i = 0; i < 3; i++)
                                for (int j = 0; j < 3; j++)
                                    tmp_mid_A(i, j) = tmp_A(i + 3, j + 3);
                            for (int i = 0; i < 3; i++)
                                tmp_mid_b[i] = b[i + 3];
                            Eigen::Vector3f tmp_increase = tmp_mid_A.ldlt().solve(tmp_mid_b);
                            for (int i = 0; i < 3; i++)
                            {
                                increase_of_ksai[i] = 0;
                                increase_of_ksai[i + 3] = tmp_increase[i];
                            }
                        }
                        else if (m_flag == 5)
                        {
                            float A_t = tmp_A(0, 0);
                            float b_t = b[0];
                            float inc = b_t / A_t;
                            increase_of_ksai[0] = inc;
                            for (int i = 1; i < 6; i++)
                                increase_of_ksai[i] = 0;
                        }
                        else if (m_flag == 6)
                        {
                            float A_t = tmp_A(1, 1);
                            float b_t = b[1];
                            float inc = b_t / A_t;
                            increase_of_ksai[1] = inc;
                            for (int i = 2; i < 6; i++)
                                increase_of_ksai[i] = 0;
                            increase_of_ksai[0] = 0;
                        }
                    }
                    else
                        increase_of_ksai = tmp_A.ldlt().solve(b);
                    pow_of_lambda++;

                    Sophus::SE3f candidate_reference_to_frame = Sophus::SE3f::exp((increase_of_ksai)) * reference_to_frame;

                    Eigen::Vector3f trans = reference_to_frame.translation();
                    if (m_flag == 5)
                    {
                        Eigen::Vector3f inc(b[0] / tmp_A(0, 0), 0, 0);
                        candidate_reference_to_frame = Sophus::SE3f(reference_to_frame.rotationMatrix(), inc + trans);
                    }
                    if (m_flag == 6)
                    {
                        Eigen::Vector3f inc(0, b[1] / tmp_A(1, 1), 0);
                        candidate_reference_to_frame = Sophus::SE3f(reference_to_frame.rotationMatrix(), inc + trans);
                    }

                    float tmp_error = calculate_error(reference, frame, candidate_reference_to_frame, level);

                    if (m_num_index < MIN_GOODPERALL_PIXEL_ABSMIN * (m_width >> level) * (m_height >> level))
                    {
                        m_diverged = true;
                        m_tracking_was_good = false;
                        lock->unlock_shared();
                        return SE3();
                    }
                    if (tmp_error < error)
                    {
                        reference_to_frame = candidate_reference_to_frame;

                        if (useAffineLightningEstimation)
                        {
                            m_affine_estimate_a = m_affine_estimate_a_last_it;
                            m_affine_estimate_b = m_affine_estimate_b_last_it;
                        }

                        if (tmp_error / error > m_hyperparameters.convergenceEps[level])
                        {
                            it = m_hyperparameters.maxItsPerLvl[level];
                        }
                        tmp_last_residual = error = tmp_error;

                        if (lambda <= 0.2)
                            lambda = 0;
                        else
                            lambda *= m_hyperparameters.lambdaSuccessFac;
                        break;
                    }
                    else
                    {

                        if (!(increase_of_ksai.dot(increase_of_ksai) > m_hyperparameters.stepSizeMin[level]))
                        {


                            it = m_hyperparameters.maxItsPerLvl[level];
                            break;
                        }
                        if (lambda == 0)
                            lambda = 0.2;
                        else
                            lambda *= std::pow(m_hyperparameters.lambdaFailFac, pow_of_lambda);
                    }
                }
            }
        }

        m_last_residual = tmp_last_residual;

        float good_in_all = m_last_good_count / (frame->get_width(SE3TRACKING_MIN_LEVEL) * frame->get_height(SE3TRACKING_MIN_LEVEL));
        float good_in_count = m_last_good_count / (m_last_good_count + m_last_bad_count);
        m_tracking_was_good = !m_diverged && (good_in_all > MIN_GOODPERALL_PIXEL) && (good_in_count > MIN_GOODPERGOODBAD_PIXEL);

        if (m_tracking_was_good)
        {
            reference->m_keyframe->m_num_frames_tracked_on_this++;
        }


        frame->m_initial_tracked_residual = m_last_residual / m_point_usage;
        frame->m_pose->m_this_to_parent_raw = sim3FromSE3(toSophus(reference_to_frame.inverse()), 1);
        frame->m_pose->m_tracking_parent = reference->m_keyframe->m_pose;

        lock->unlock_shared();

        return toSophus(reference_to_frame.inverse());
    }

    float c_SE3Tracker::calculate_error(c_TrackingReference* reference, c_Frame* frame, const Sophus::SE3f& reference_to_frame, int level)
    {
        Eigen::Vector3f* reference_point_pos_buffer = reference->m_pos_data[level];
        Eigen::Vector2f* reference_color_and_var_buffer = reference->m_color_and_var_data[level];
        int num = reference->m_num_data[level];
        int w = frame->get_width(level);
        int h = frame->get_height(level);
        Eigen::Matrix3f K_level = frame->get_K(level);
        float fx_level = K_level(0, 0);
        float fy_level = K_level(1, 1);
        float cx_level = K_level(0, 2);
        float cy_level = K_level(1, 2);
        Eigen::Matrix3f rotation = reference_to_frame.rotationMatrix();
        Eigen::Vector3f trans = reference_to_frame.translation();
        const Eigen::Vector4f* frame_gradients = frame->get_gradients(level);

        int index = 0;
        float tmp_sum_error = 0;

        int* index_buffer_buffer = level == SE3TRACKING_MIN_LEVEL ? reference->m_point_pos_in_xy_grid[level] : 0;
        bool* is_good_out_buffer = index_buffer_buffer != 0 ? frame->get_ref_pixel_was_good() : 0;

        int good_count = 0, bad_count = 0;
        float sum_signed_res = 0;

        float sxx = 0, syy = 0, sx = 0, sy = 0, sw = 0;
        float usage_count = 0;

        //std::string filename = "/home/g107904/my_code/data/my.txt";
        //FILE* fp = fopen(filename.c_str(),"w+");

        for (int i = 0; i < num; i++)
        {
            Eigen::Vector3f reference_point_pos = reference_point_pos_buffer[i];
            Eigen::Vector2f reference_color_and_var = reference_color_and_var_buffer[i];

            Eigen::Vector3f cur_point_pos = rotation * (reference_point_pos)+trans;
            float u_new = (cur_point_pos[0] / cur_point_pos[2]) * fx_level + cx_level;
            float v_new = (cur_point_pos[1] / cur_point_pos[2]) * fy_level + cy_level;

            if (!(u_new > 1 && v_new > 1 && u_new < w - 2 && v_new < h - 2))
            {
                if (is_good_out_buffer != 0)
                {
                    is_good_out_buffer[index_buffer_buffer[i]] = false;
                }
                continue;
            }

            Eigen::Vector4f frame_gradient_and_color = get_interpolated_element<Eigen::Vector4f>(frame_gradients, u_new, v_new, w);

            float residual_reference = m_affine_estimate_a * (reference_color_and_var)[0] + m_affine_estimate_b;
            float residual_frame = frame_gradient_and_color[2];
            float residual = residual_reference - residual_frame;

            float weight = fabsf(residual) < 5.0f ? 1 : 5.0f / fabsf(residual);
            sxx += residual_reference * residual_reference * weight;
            syy += residual_frame * residual_frame * weight;
            sx += residual_reference * weight;
            sy += residual_frame * weight;
            sw += weight;

            float residual_threshold =
                MAX_DIFF_CONSTANT
                + MAX_DIFF_GRAD_MULT
                * (frame_gradient_and_color[0] * frame_gradient_and_color[0]
                    + frame_gradient_and_color[1] * frame_gradient_and_color[1]);

            bool is_good = residual * residual / residual_threshold < 1;
            if (is_good_out_buffer != 0)
            {
                is_good_out_buffer[index_buffer_buffer[i]] = is_good;
            }

            float cur_x_hat = cur_point_pos[0];
            float cur_y_hat = cur_point_pos[1];
            float cur_z_hat = cur_point_pos[2];
            float cur_d = 1.0f / (reference_point_pos)[2];
            float cur_rp = residual;
            float cur_dx = frame_gradient_and_color[0];
            float cur_dy = frame_gradient_and_color[1];
            float cur_gx = fx_level * frame_gradient_and_color[0];
            float cur_gy = fy_level * frame_gradient_and_color[1];
            float cur_sigma_d = (reference_color_and_var)[1];
            float cur_tx = trans[0];
            float cur_ty = trans[1];
            float cur_tz = trans[2];




            float cur_patial_rp_div_patial_d =
                cur_gx * ((cur_tx * cur_z_hat - cur_tz * cur_x_hat) / (cur_z_hat * cur_z_hat * cur_d))
                + cur_gy * ((cur_ty * cur_z_hat - cur_tz * cur_y_hat) / (cur_z_hat * cur_z_hat * cur_d));

            //my_rot
            Eigen::Vector3f mid_pos = rotation.transpose() * reference_point_pos;
            Eigen::Vector3f mid_tmp;
            float z_j_frac = 1.0f / cur_point_pos[2];
            float z_j_frac2 = z_j_frac * z_j_frac;
            mid_tmp[0] = cur_gx * z_j_frac;
            mid_tmp[1] = cur_gy * z_j_frac;
            mid_tmp[2] = -cur_gx * cur_point_pos[0] * z_j_frac2 - cur_gy * cur_point_pos[1] * z_j_frac2;
            cur_patial_rp_div_patial_d = -mid_tmp.dot(mid_pos) * cur_d;

            float sigma_rp = ((cameraPixelNoise2)+cur_sigma_d * cur_patial_rp_div_patial_d * cur_patial_rp_div_patial_d);
            /*
                float r_abs = fabs(cur_rp/sqrt(sigma_rp));
                float tmp_error;
                if(r_abs < (m_hyperparameters.huber_d/2))
                {
                    tmp_error = cur_rp*cur_rp / sigma_rp;
                }
                else
                {
                    tmp_error = fabs(m_hyperparameters.huber_d/2/r_abs)*cur_rp*cur_rp/sigma_rp;
                }
            */
            float w_p = 1.0f / sigma_rp;

            float weighted_rp = fabs(cur_rp * sqrtf(w_p));

            float wh = fabs(weighted_rp < (m_hyperparameters.huber_d / 2) ? 1 : (m_hyperparameters.huber_d / 2) / weighted_rp);

            float tmp_error = wh * w_p * cur_rp * cur_rp;

            tmp_sum_error += tmp_error;

            //fprintf(fp,"%d %.8f\n",index,tmp_sum_error);
            //*(m_frame_residual_var+index) = sigma_rp;


            {
                m_frame_point_pos_x[index] = cur_point_pos[0];
                m_frame_point_pos_y[index] = cur_point_pos[1];
                m_frame_point_pos_z[index] = cur_point_pos[2];
                m_frame_gradient_dx[index] = cur_gx;
                m_frame_gradient_dy[index] = cur_gy;
                m_frame_idepth[index] = cur_d;
                m_frame_idepth_var[index] = cur_sigma_d;
                m_frame_residual[index] = residual;

                m_frame_residual_var[index] = wh * w_p;

                index++;

                if (is_good)
                {
                    sum_signed_res += residual;
                    good_count++;
                }
                else
                    bad_count++;

                float depth_change = (reference_point_pos)[2] / cur_point_pos[2];
                usage_count += depth_change < 1 ? depth_change : 1;
            }
        }
        m_num_index = index;
        //fclose(fp);

        m_point_usage = usage_count / num;
        m_last_good_count = good_count;
        m_last_bad_count = bad_count;
        m_last_mean_residual = sum_signed_res / good_count;

        m_affine_estimate_a_last_it = sqrtf((syy - sy * sy / sw) / (sxx - sx * sx / sw));
        m_affine_estimate_b_last_it = (sy - m_affine_estimate_a_last_it * sx) / sw;

        return tmp_sum_error / (float)m_num_index;
    }

    void c_SE3Tracker::calculate_iterate(Matrix6f& A, Vector6f& b, float fx_level, float fy_level)
    {
        A = Matrix6f::Zero();
        b = Vector6f::Zero();
        for (int i = 0; i < m_num_index; i++)
        {
            float x_hat = *(m_frame_point_pos_x + i);
            float y_hat = *(m_frame_point_pos_y + i);
            float z_hat = *(m_frame_point_pos_z + i);
            float rp = *(m_frame_residual + i);
            float gx = *(m_frame_gradient_dx + i);
            float gy = *(m_frame_gradient_dy + i);
            //float sigma_rp = *(m_frame_residual_var+i);
            float w_p = *(m_frame_residual_var + i);

            //for the same precision
            float z = 1.0f / z_hat;
            float z_sqr = 1.0f / (z_hat * z_hat);

            Vector6f J;
            J[0] = gx * z;
            //J[0] = 0;
            J[1] = gy * z;
            //J[1] = 0;
            J[2] = gx * (-x_hat * z_sqr) + gy * (-y_hat * z_sqr);
            if (m_flag)
            {
                if (m_flag < 3)
                {
                    J[m_flag - 1] = 0;
                    J[2] = 0;
                }
                else
                {
                    J[0] = J[1] = J[2] = 0;
                }
            }
            J[3] = gx * (-x_hat * y_hat * z_sqr) + gy * (-(1.0 + y_hat * y_hat * z_sqr));
            J[4] = gx * (1.0 + x_hat * x_hat * z_sqr) + gy * (x_hat * y_hat * z_sqr);
            J[5] = gx * (-y_hat * z) + gy * (x_hat * z);

            float weight = w_p;
            Matrix6f tmp_mid_A = J * J.transpose();
            A = A + weight * tmp_mid_A;
            b = b + weight * rp * J;
            /*
            if(i == 0)
            {
                        for(int j = 0;j < 6;j++)
                    for(int k = 0;k < 6;k++)
                        A(j,k) = weight*tmp_mid_A(j,k);
                    for(int j = 0;j < 6;j++)
                    b[j] = weight*rp*J[j];
                continue;
            }
                for(int j = 0;j < 6;j++)
            for(int k = 0;k < 6;k++)
                A(j,k) = A(j,k) + weight*tmp_mid_A(j,k);
            for(int j = 0;j < 6;j++)
            b[j] = b[j] + weight*rp*J[j];
            */
        }

        A /= (float)m_num_index;
        b /= (float)m_num_index;
        b = b;

    }

    SE3 c_SE3Tracker::trackFrame_SO3(c_TrackingReference* reference, c_Frame* frame, const SE3& initial)
    {
        c_read_write_lock* lock = frame->get_active_lock();
        Sophus::SE3f reference_to_frame = initial.inverse().cast<float>();

        m_diverged = false;
        m_tracking_was_good = true;
        m_affine_estimate_a = 1; m_affine_estimate_b = 0;

        float tmp_last_residual = 0;

        for (int level = SE3TRACKING_MAX_LEVEL - 1; level >= SE3TRACKING_MIN_LEVEL; level--)
        {
            reference->make_pointcloud(level);
            float error = calculate_error_SO3(reference, frame, reference_to_frame, level);
            float lambda = m_hyperparameters.lambdaInitial[level];
            if (m_num_index < MIN_GOODPERALL_PIXEL_ABSMIN * (m_width >> level) * (m_height >> level))
            {
                m_diverged = true;
                m_tracking_was_good = false;

                lock->unlock_shared();
                return SE3();
            }

            if (useAffineLightningEstimation)
            {
                m_affine_estimate_a = m_affine_estimate_a_last_it;
                m_affine_estimate_b = m_affine_estimate_b_last_it;
            }

            for (int it = 0; it < m_hyperparameters.maxItsPerLvl[level]; it++)
            {
                Vector6f b;
                Matrix6f A;
                Eigen::Matrix3f K_level = frame->get_K(level);
                float fx_level = K_level(0, 0);
                float fy_level = K_level(1, 1);
                calculate_iterate_SO3(A, b, fx_level, fy_level);

                int pow_of_lambda = 0;
                while (true)
                {
                    Matrix6f tmp_A = A;
                    for (int i = 0; i < 6; i++)
                        tmp_A(i, i) *= (1 + lambda);
                    Vector6f increase_of_ksai = tmp_A.ldlt().solve(b);
                    pow_of_lambda++;

                    Sophus::SE3f candidate_reference_to_frame = Sophus::SE3f::exp((increase_of_ksai)) * reference_to_frame;
                    float tmp_error = calculate_error_SO3(reference, frame, candidate_reference_to_frame, level);

                    if (m_num_index < MIN_GOODPERALL_PIXEL_ABSMIN * (m_width >> level) * (m_height >> level))
                    {
                        m_diverged = true;
                        m_tracking_was_good = false;
                        lock->unlock_shared();
                        return SE3();
                    }
                    if (tmp_error < error)
                    {
                        reference_to_frame = candidate_reference_to_frame;

                        if (useAffineLightningEstimation)
                        {
                            m_affine_estimate_a = m_affine_estimate_a_last_it;
                            m_affine_estimate_b = m_affine_estimate_b_last_it;
                        }

                        if (tmp_error / error > m_hyperparameters.convergenceEps[level])
                        {
                            it = m_hyperparameters.maxItsPerLvl[level];
                        }
                        tmp_last_residual = error = tmp_error;

                        if (lambda <= 0.2)
                            lambda = 0;
                        else
                            lambda *= m_hyperparameters.lambdaSuccessFac;
                        break;
                    }
                    else
                    {

                        if (!(increase_of_ksai.dot(increase_of_ksai) > m_hyperparameters.stepSizeMin[level]))
                        {


                            it = m_hyperparameters.maxItsPerLvl[level];
                            break;
                        }
                        if (lambda == 0)
                            lambda = 0.2;
                        else
                            lambda *= std::pow(m_hyperparameters.lambdaFailFac, pow_of_lambda);
                    }
                }
            }
        }

        m_last_residual = tmp_last_residual;

        float good_in_all = m_last_good_count / (frame->get_width(SE3TRACKING_MIN_LEVEL) * frame->get_height(SE3TRACKING_MIN_LEVEL));
        float good_in_count = m_last_good_count / (m_last_good_count + m_last_bad_count);
        m_tracking_was_good = !m_diverged && (good_in_all > MIN_GOODPERALL_PIXEL) && (good_in_count > MIN_GOODPERGOODBAD_PIXEL);

        if (m_tracking_was_good)
        {
            reference->m_keyframe->m_num_frames_tracked_on_this++;
        }

        /*
        frame->m_initial_tracked_residual = m_last_residual / m_point_usage;
        frame->m_pose->m_this_to_parent_raw = sim3FromSE3(toSophus(reference_to_frame.inverse()), 1);
        frame->m_pose->m_tracking_parent = reference->m_keyframe->m_pose;
        */
        lock->unlock_shared();

        return toSophus(reference_to_frame.inverse());
    }

    float c_SE3Tracker::calculate_error_SO3(c_TrackingReference* reference, c_Frame* frame, const Sophus::SE3f& reference_to_frame, int level)
    {
        Eigen::Vector3f* reference_point_pos = reference->m_pos_data[level];
        Eigen::Vector2f* reference_color_and_var = reference->m_color_and_var_data[level];
        int num = reference->m_num_data[level];
        int w = frame->get_width(level);
        int h = frame->get_height(level);
        Eigen::Matrix3f K_level = frame->get_K(level);
        float fx_level = K_level(0, 0);
        float fy_level = K_level(1, 1);
        float cx_level = K_level(0, 2);
        float cy_level = K_level(1, 2);
        Eigen::Matrix3f rotation = reference_to_frame.rotationMatrix();
        Eigen::Vector3f trans = reference_to_frame.translation();
        const Eigen::Vector4f* frame_gradients = frame->get_gradients(level);

        int index = 0;
        float tmp_sum_error = 0;

        int* index_buffer = level == SE3TRACKING_MIN_LEVEL ? reference->m_point_pos_in_xy_grid[level] : 0;
        bool* is_good_out_buffer = index_buffer != 0 ? frame->get_ref_pixel_was_good() : 0;

        int good_count = 0, bad_count = 0;
        float sum_signed_res = 0;

        float sxx = 0, syy = 0, sx = 0, sy = 0, sw = 0;
        float usage_count = 0;

        //std::string filename = "/home/g107904/my_code/data/my.txt";
        //FILE* fp = fopen(filename.c_str(),"w+");

        for (int i = 0; i < num; reference_point_pos++, reference_color_and_var++, index_buffer++, i++)
        {
            Eigen::Vector3f cur_point_pos = rotation * (*reference_point_pos) + trans;
            float u_new = (cur_point_pos[0] / cur_point_pos[2]) * fx_level + cx_level;
            float v_new = (cur_point_pos[1] / cur_point_pos[2]) * fy_level + cy_level;

            if (!(u_new > 1 && v_new > 1 && u_new < w - 2 && v_new < h - 2))
            {
                if (is_good_out_buffer != 0)
                {
                    is_good_out_buffer[*index_buffer] = false;
                }
                continue;
            }

            Eigen::Vector4f frame_gradient_and_color = get_interpolated_element<Eigen::Vector4f>(frame_gradients, u_new, v_new, w);

            float residual_reference = m_affine_estimate_a * (*reference_color_and_var)[0] + m_affine_estimate_b;
            float residual_frame = frame_gradient_and_color[2];
            float residual = residual_reference - residual_frame;

            *(m_frame_point_pos_x + index) = cur_point_pos[0];
            *(m_frame_point_pos_y + index) = cur_point_pos[1];
            *(m_frame_point_pos_z + index) = cur_point_pos[2];
            *(m_frame_gradient_dx + index) = fx_level * frame_gradient_and_color[0];
            *(m_frame_gradient_dy + index) = fy_level * frame_gradient_and_color[1];
            *(m_frame_idepth + index) = 1.0f / (*reference_point_pos)[2];
            *(m_frame_idepth_var + index) = (*reference_color_and_var)[1];
            *(m_frame_residual + index) = residual;

            float weight = fabsf(residual) < 5.0f ? 1 : 5.0f / fabsf(residual);
            sxx += residual_reference * residual_reference * weight;
            syy += residual_frame * residual_frame * weight;
            sx += residual_reference * weight;
            sy += residual_frame * weight;
            sw += weight;

            float residual_threshold =
                MAX_DIFF_CONSTANT
                + MAX_DIFF_GRAD_MULT
                * (frame_gradient_and_color[0] * frame_gradient_and_color[0]
                    + frame_gradient_and_color[1] * frame_gradient_and_color[1]);

            bool is_good = residual * residual / residual_threshold < 1;
            if (is_good_out_buffer != 0)
            {
                is_good_out_buffer[*index_buffer] = is_good;
            }

            float cur_x_hat = cur_point_pos[0];
            float cur_y_hat = cur_point_pos[1];
            float cur_z_hat = cur_point_pos[2];
            float cur_d = 1.0f / (*reference_point_pos)[2];
            float cur_rp = residual;
            float cur_dx = frame_gradient_and_color[0];
            float cur_dy = frame_gradient_and_color[1];
            float cur_gx = *(m_frame_gradient_dx + index);
            float cur_gy = *(m_frame_gradient_dy + index);
            float cur_sigma_d = (*reference_color_and_var)[1];
            float cur_tx = trans[0];
            float cur_ty = trans[1];
            float cur_tz = trans[2];

            float cur_patial_rp_div_patial_d =
                cur_gx * ((cur_tx * cur_z_hat - cur_tz * cur_x_hat) / (cur_z_hat * cur_z_hat * cur_d))
                + cur_gy * ((cur_ty * cur_z_hat - cur_tz * cur_y_hat) / (cur_z_hat * cur_z_hat * cur_d));

            float sigma_rp = ((cameraPixelNoise2)+cur_sigma_d * cur_patial_rp_div_patial_d * cur_patial_rp_div_patial_d);
            /*
                float r_abs = fabs(cur_rp/sqrt(sigma_rp));
                float tmp_error;
                if(r_abs < (m_hyperparameters.huber_d/2))
                {
                    tmp_error = cur_rp*cur_rp / sigma_rp;
                }
                else
                {
                    tmp_error = fabs(m_hyperparameters.huber_d/2/r_abs)*cur_rp*cur_rp/sigma_rp;
                }
            */
            float w_p = 1.0f / sigma_rp;

            float weighted_rp = fabs(cur_rp * sqrtf(w_p));

            float wh = fabs(weighted_rp < (m_hyperparameters.huber_d / 2) ? 1 : (m_hyperparameters.huber_d / 2) / weighted_rp);

            float tmp_error = wh * w_p * cur_rp * cur_rp;

            tmp_sum_error += tmp_error;

            //fprintf(fp,"%d %.8f\n",index,tmp_sum_error);
            //*(m_frame_residual_var+index) = sigma_rp;
            *(m_frame_residual_var + index) = wh * w_p;
            index++;

            if (is_good)
            {
                sum_signed_res += residual;
                good_count++;
            }
            else
                bad_count++;

            float depth_change = (*reference_point_pos)[2] / cur_point_pos[2];
            usage_count += depth_change < 1 ? depth_change : 1;
        }
        m_num_index = index;
        //fclose(fp);

        m_point_usage = usage_count / num;
        m_last_good_count = good_count;
        m_last_bad_count = bad_count;
        m_last_mean_residual = sum_signed_res / good_count;

        m_affine_estimate_a_last_it = sqrtf((syy - sy * sy / sw) / (sxx - sx * sx / sw));
        m_affine_estimate_b_last_it = (sy - m_affine_estimate_a_last_it * sx) / sw;

        return tmp_sum_error / (float)m_num_index;
    }

    void c_SE3Tracker::calculate_iterate_SO3(Matrix6f& A, Vector6f& b, float fx_level, float fy_level)
    {

        for (int i = 0; i < m_num_index; i++)
        {
            float x_hat = *(m_frame_point_pos_x + i);
            float y_hat = *(m_frame_point_pos_y + i);
            float z_hat = *(m_frame_point_pos_z + i);
            float rp = *(m_frame_residual + i);
            float gx = *(m_frame_gradient_dx + i);
            float gy = *(m_frame_gradient_dy + i);
            //float sigma_rp = *(m_frame_residual_var+i);
            float w_p = *(m_frame_residual_var + i);

            //for the same precision
            float z = 1.0f / z_hat;
            float z_sqr = 1.0f / (z_hat * z_hat);

            Vector6f J;
            J[0] = 0;
            J[1] = 0;
            J[2] = 0;
            J[3] = gx * (-x_hat * y_hat * z_sqr) + gy * (-(1.0 + y_hat * y_hat * z_sqr));
            J[4] = gx * (1.0 + x_hat * x_hat * z_sqr) + gy * (x_hat * y_hat * z_sqr);
            J[5] = gx * (-y_hat * z) + gy * (x_hat * z);

            /*
                float r_abs = fabs(rp/sqrt(sigma_rp));
                float weight;
                if(r_abs < (m_hyperparameters.huber_d/2))
                {
                    weight = 1 / sigma_rp;
                }
                else
                {
                    weight = fabs(m_hyperparameters.huber_d/2/r_abs)/sigma_rp;
                }
            */
            float weight = w_p;
            Matrix6f tmp_mid_A = J * J.transpose();
            //A = A + J*J.transpose()*weight;
            //b = b - J*(rp*weight);
            if (i == 0)
            {
                for (int j = 0; j < 6; j++)
                    for (int k = 0; k < 6; k++)
                        A(j, k) = weight * tmp_mid_A(j, k);
                for (int j = 0; j < 6; j++)
                    b[j] = weight * rp * J[j];
                continue;
            }
            for (int j = 0; j < 6; j++)
                for (int k = 0; k < 6; k++)
                    A(j, k) = A(j, k) + weight * tmp_mid_A(j, k);
            for (int j = 0; j < 6; j++)
                b[j] = b[j] + weight * rp * J[j];
        }
        A /= (float)m_num_index;
        b /= (float)m_num_index;
        b = b;
    }

}
