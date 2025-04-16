#pragma once
#include "util/settings.h"
#include "util/EigenCoreInclude.h"
#include "util/Sophus_util.h"
#include <mutex>

typedef Eigen::Matrix<float, 7, 7> Matrix7f;
typedef Eigen::Matrix<float, 7, 1> Vector7f;
namespace lsd_slam
{
    class c_TrackingReference;
    class c_Frame;

    struct c_sim3_residual
    {
        float m_sum_res_depth;
        float m_sum_res_photometric;
        int m_num_depth;
        int m_num_photometric;

        float m_mean_depth;
        float m_mean_photometric;
        float m_mean;

        inline c_sim3_residual()
        {
            m_mean_depth = 0;
            m_mean_photometric = 0;
            m_num_depth = m_num_photometric = m_sum_res_depth = m_sum_res_photometric = 0;
            
        }
    };

    class c_Sim3Tracker
    {
        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW

            int m_width,m_height;

            Eigen::Matrix3f m_K,m_KInv;
            float m_fx,m_fy,m_cx,m_cy;
            float m_fxi,m_fyi,m_cxi,m_cyi;

            DenseDepthTrackerSettings m_hyperparameters;

            Matrix7f m_last_sim3_hessian;

            float m_point_usage;

            float m_last_residual,m_last_depth_residual,m_last_photometric_residual;
            float m_last_residual_unweighted,m_last_depth_residual_unweighted,m_last_photometric_residual_unweighted;
            
            float m_affine_trans_a;
            float m_affine_trans_b;
            
            bool m_diverged;

            c_Sim3Tracker(int w,int h,Eigen::Matrix3f K);
            c_Sim3Tracker(const c_Sim3Tracker&) = delete;
            c_Sim3Tracker& operator = (const c_Sim3Tracker&) = delete;
            ~c_Sim3Tracker();
            
            Sim3 trackFrame(
                c_TrackingReference* reference,c_Frame* frame,const Sim3& frame_to_reference_initial,int start_level,int end_level);

            c_sim3_residual calculate_error(c_TrackingReference* reference,c_Frame* frame,Sim3& reference_to_frame,int level);

            void calculate_iterate(Matrix7f& A,Vector7f& b,float fx_level,float fy_level);
        private:
            float* m_frame_point_pos_x;
	        float* m_frame_point_pos_y;
	        float* m_frame_point_pos_z;
            float* m_frame_gradient_dx;
            float* m_frame_gradient_dy;

            float* m_residual_depth;
            float* m_residual_photometric;

            float* m_weighted_depth;
            float* m_weighted_photometric;
            int m_num_index;

            float affine_trans_a_last_it;
            float affine_trans_b_last_it;
    };


}
