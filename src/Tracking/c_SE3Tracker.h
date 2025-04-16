#pragma once
#include "util/settings.h"
#include "util/EigenCoreInclude.h"
#include "util/Sophus_util.h"
#include <mutex>

typedef Eigen::Matrix<float, 6, 6> Matrix6f;
typedef Eigen::Matrix<float, 6, 1> Vector6f;
namespace lsd_slam
{
    class c_TrackingReference;
    class c_Frame;
    
    class c_SE3Tracker
    {
    public:

        int m_width,m_height;
        Eigen::Matrix3f m_K;
        DenseDepthTrackerSettings m_hyperparameters;

        float m_point_usage;

        float m_last_residual;
        float m_last_mean_residual;

        float m_last_good_count;
        float m_last_bad_count;
        
        
        float m_affine_estimate_a;
        float m_affine_estimate_b;

        bool m_diverged;
        bool m_tracking_was_good;

        c_SE3Tracker(int w, int h, Eigen::Matrix3f K);
        ~c_SE3Tracker();

        float check_overlap(c_Frame* reference,SE3& reference_to_frame);

        SE3 quick_trackFrame(c_Frame* reference,c_Frame* frame,SE3& reference_to_frame);
        float quick_calculate_error(c_Frame* reference,c_Frame* frame,const Sophus::SE3f& reference_to_frame);
        
        SE3 trackFrame(c_TrackingReference* reference,c_Frame* frame,const SE3& frameToReference_initialEstimate);
        float calculate_error(c_TrackingReference* reference,c_Frame* frame,const Sophus::SE3f& reference_to_frame,int level);
        void calculate_iterate(Matrix6f& A,Vector6f& b,float fx_level,float fy_level);
    private:
        float* m_frame_point_pos_x;
	    float* m_frame_point_pos_y;
	    float* m_frame_point_pos_z;
        float* m_frame_gradient_dx;
        float* m_frame_gradient_dy;
        float* m_frame_idepth;
	    float* m_frame_idepth_var;
        float* m_frame_residual;
        float* m_frame_residual_var;
        int m_num_index;

        float m_affine_estimate_a_last_it;
        float m_affine_estimate_b_last_it;
    };
}
