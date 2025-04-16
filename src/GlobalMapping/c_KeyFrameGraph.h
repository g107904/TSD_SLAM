#pragma once
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <mutex>
#include "util/EigenCoreInclude.h"
#include <Eigen/StdVector>
#include <g2o/core/sparse_optimizer.h>
#include "util/Sophus_util.h"
#include "util/settings.h"
#include "deque"
#include "util/c_read_write_lock.h"
#include <memory>

namespace lsd_slam
{
    class c_Frame;
    class c_KeyFrameGraph;
    class c_vertex_sim3;
    class c_edge_sim3;
    class c_FramePose;
    class c_SE3Tracker;

    struct c_KFConstraintStruct
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        c_Frame* m_first_frame;
        c_Frame* m_second_frame;
        Sophus::Sim3d m_second_to_first;
        Eigen::Matrix<double,7,7> m_information;
        g2o::RobustKernel* m_robustKernel;
        c_edge_sim3* m_edge;

        float m_usage;
        float m_mean_depth_residual;
        float m_mean_photometric_residual;
        float m_mean_residual;

        float m_reciprocal_consistency;
    
        int m_index_in_all_edges;

        inline c_KFConstraintStruct()
        {
            m_first_frame = m_second_frame = 0;
            m_information.setZero();
            m_robustKernel = 0;
            m_edge = 0;

            m_usage = m_mean_depth_residual = m_mean_photometric_residual = m_mean_residual = 0;

            m_reciprocal_consistency = 0;

            m_index_in_all_edges = -1;
        }

        ~c_KFConstraintStruct();
    };

    struct c_TrackableKFStruct
    {
        c_Frame* m_reference;
        SE3 m_reference_to_frame;
        float m_distance;
        float m_angle;
    };

    class c_KeyFrameGraph
    {
        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

            c_KeyFrameGraph(int w,int h,Eigen::Matrix3f K);

            ~c_KeyFrameGraph();

            //add KF to buffer
            void add_keyframe(c_Frame* frame);

            //add frame pose to m_all_frame_poses
            void add_frame(c_Frame* frame);

            //trans constraint to edge, then add to m_edges_all
            void insert_constraint(c_KFConstraintStruct* constraint);

            //use g2o optimize
            int optimize(int num_iterations);

            //add buffer data(KF,edge) to graph
            bool add_elements_from_buffer();

            //get a hashmap of the distance between frame and cur_frame
            void calculate_graph_distances_to_frame(c_Frame* frame,std::unordered_map<c_Frame*,int>* distance_map);

            //---------------------------used for searching trackable keyframe--------------------//

            inline float get_reference_frame_score(float distance_squared,float usage)
            {
                return distance_squared * KFDistWeight * KFDistWeight + (1-usage)*(1-usage)*KFUsageWeight*KFUsageWeight;
            }

            //function of getting some trackable frames
            void find_euclidean_overlap_frames(
                c_Frame* frame,float distance_threshold,float angle_threshold,
                bool check_both_scales,
                std::vector<c_TrackableKFStruct,Eigen::aligned_allocator<c_TrackableKFStruct> >& potential_reference_frames
            ); 

            //function of finding a set of candidate frames
            void find_candidates(
                c_Frame* keyframe,float closeness_threshold,
                std::unordered_set<c_Frame*,std::hash<c_Frame*>,std::equal_to<c_Frame*>,Eigen::aligned_allocator<c_Frame*> >& results
            );

            //function of finding the best candidate frame 
            c_Frame* find_reposition_candidate(c_Frame* frame,float max_score = 1);

            //--------------------------used for searching trackable keyframe----------------------//


            //merge optimization(call in depth map  estimation thread)
            void merge_optimization_result();

            c_Frame* find_appearance_based_candidate(c_Frame* keyframe);


            // contains all finished keyframes
            c_read_write_lock m_keyframes_all_mutex;

            //std::vector<c_Frame*,Eigen::aligned_allocator<c_Frame*> > m_keyframes_all;
            std::vector<c_Frame*> m_keyframes_all;

            //id to KF map
            c_read_write_lock m_id_to_keyframe_mutex;
            std::unordered_map<int,std::shared_ptr<c_Frame>,std::hash<int>,std::equal_to<int>,Eigen::aligned_allocator< std::pair<const int,std::shared_ptr<c_Frame> > > > m_id_to_keyframe;

            //contains all edges(constraints)
            c_read_write_lock m_edges_lists_mutex;

            //std::vector<c_KFConstraintStruct*,Eigen::aligned_allocator<c_KFConstraintStruct*> > m_edges_all;
            std::vector<c_KFConstraintStruct* > m_edges_all;

            //contrains all frame poses
            c_read_write_lock m_all_frame_poses_mutex;
            //std::vector<c_FramePose*,Eigen::aligned_allocator<c_FramePose*> > m_all_frame_poses;
            std::vector<c_FramePose* > m_all_frame_poses;

            //contrains all graph keyframes
            std::mutex m_keyframes_for_retrack_mutex;
            std::deque<c_Frame*> m_keyframes_for_retrack;

            //-------------for thread safety-----------------------
            c_read_write_lock m_pose_consistency_mutex;
            bool m_have_unmerged_optimization;

            std::deque<c_Frame*> m_new_keyframes;
            std::condition_variable m_new_keyframes_created_signal;
            std::mutex m_new_keyframes_mutex;
            //-----------------------------------------------------



            //debug
            void print_all_pose();
        private:

            g2o::SparseOptimizer m_graph;

            //KF buffer 
            //std::vector<c_Frame*,Eigen::aligned_allocator<c_Frame*> > m_new_keyframes_buffer;
            std::vector<c_Frame*> m_new_keyframes_buffer;
            
            //constraint(contains edge) buffer
            //std::vector<c_KFConstraintStruct*,Eigen::aligned_allocator<c_KFConstraintStruct*> > m_new_edge_buffer;
            std::vector<c_KFConstraintStruct*> m_new_edge_buffer;

            //count for edge id
            int m_next_edge_id;

            c_SE3Tracker* m_se3_tracker;

            //used for calculating angle 
            float m_fowx,m_fowy;

    };
}
