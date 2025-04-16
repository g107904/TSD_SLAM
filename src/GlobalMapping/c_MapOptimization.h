#pragma once
#include <vector>
#include <mutex>
#include <condition_variable>

#include "util/settings.h"
#include "util/Sophus_util.h"

namespace lsd_slam
{
    class c_KeyFrameGraph;
    class c_SE3Tracker;
    class c_Sim3Tracker;
    class c_Frame;
    class c_FramePose;
    class c_KFConstraintStruct;
    class c_TrackingReference;
    
    class c_MapOptimization
    {
        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

            int m_width,m_height;
            Eigen::Matrix3f m_K;

            int m_failed_to_retrack;
            
            c_KeyFrameGraph* m_keyframe_graph;
            c_SE3Tracker* m_se3_constraint_tracker;
            c_Sim3Tracker* m_sim3_constraint_tracker;

            c_TrackingReference* m_new_KF_tracking_reference;
            c_TrackingReference* m_candidate_tracking_reference;

            bool m_new_constraint_added;
            std::mutex m_new_constraint_mutex;
            std::condition_variable m_new_constraint_created_signal;


            std::mutex m_g2o_graph_access_mutex;

            //----------------------finalize--------------
            bool m_do_full_reconstraint_track;
            int m_last_num_constraints_added_on_full_retrack;
            bool m_do_final_optimization;

            c_MapOptimization(int w,int h,Eigen::Matrix3f K);
            ~c_MapOptimization();

            void do_constraint_search(std::unique_lock<std::mutex>& lock);

            int find_constraints_for_new_KF(
                c_Frame* frame,bool force_parent = true,
                float close_candidates_threshold = 1.0);

            void test_constraint(
                c_Frame* candidate,
                c_KFConstraintStruct* &edge_1,c_KFConstraintStruct* &edge_2,
                Sim3& candidate_to_frame_init,
                float strictness);

            float try_track_sim3(
                c_TrackingReference* A,c_TrackingReference* B,
                int start_level,int end_level,
                Sim3& A_to_B,Sim3& B_to_A,
                c_KFConstraintStruct* edge_1 = 0,c_KFConstraintStruct* edge_2 = 0);
            
            


            bool do_optimization_it(int iteration_per_try,float min_change);

            void do_optimization();

    };
}
