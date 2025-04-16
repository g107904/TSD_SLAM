#include "GlobalMapping/c_KeyFrameGraph.h"
#include "DataStructures/c_Frame.h"

#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/pcg/linear_solver_pcg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/solver.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/estimate_propagator.h>
#include <g2o/core/sparse_optimizer_terminate_action.h>

#include <g2o/types/sim3/sim3.h>

#include "GlobalMapping/g2o_with_type_sim3.h"

#include "Tracking/c_SE3Tracker.h"

#include<queue>
#include<iostream>
#include<stdio.h>

#include"util/interpolation.h"


namespace lsd_slam
{
    c_KFConstraintStruct::~c_KFConstraintStruct()
    {
        if(m_edge != 0)
            delete m_edge;
    }

    c_KeyFrameGraph::c_KeyFrameGraph(int w,int h,Eigen::Matrix3f K):m_next_edge_id(0)
    {

        m_se3_tracker = new c_SE3Tracker(w,h,K);

        m_fowx = 2 * atanf((float) (w/K(0,0)) / 2.0f);
        m_fowy = 2 * atanf((float) (h/K(1,1)) / 2.0f);

        typedef g2o::BlockSolver_7_3 g2o_blocksolver;
        typedef g2o::LinearSolverCSparse<g2o_blocksolver::PoseMatrixType> linear_solver;

        linear_solver* solver = new linear_solver();

        g2o_blocksolver* blocksolver = new g2o_blocksolver(solver);

        g2o::OptimizationAlgorithmLevenberg* algorithm = new g2o::OptimizationAlgorithmLevenberg(blocksolver);

        m_graph.setAlgorithm(algorithm);

        m_graph.setVerbose(false);

        solver->setWriteDebug(true);

        blocksolver->setWriteDebug(true);

        algorithm->setWriteDebug(true);

        m_have_unmerged_optimization = false;

    }

    c_KeyFrameGraph::~c_KeyFrameGraph()
    {   
        for(c_KFConstraintStruct* edge : m_new_edge_buffer)
        {
            delete edge;
        }
        

        m_id_to_keyframe.clear();

        for(c_FramePose* pose : m_all_frame_poses)
        {
            delete pose;
        }

        delete m_se3_tracker;
    }

    void c_KeyFrameGraph::add_keyframe(c_Frame* frame)
    {
        if(frame->m_pose->m_vertex != nullptr)
        {
            return;
        }

        c_vertex_sim3* vertex = new c_vertex_sim3();
        vertex->setId(frame->get_id());

        Sophus::Sim3d cam_to_world_estimate = frame->get_scaled_cam_to_world();

        if(!frame->has_tracking_parent())
        {
            vertex->setFixed(true);
        }

        vertex->setEstimate(cam_to_world_estimate);
        vertex->setMarginalized(false);

        frame->m_pose->m_vertex = vertex;

        m_new_keyframes_buffer.push_back(frame);
    }

    void c_KeyFrameGraph::add_frame(c_Frame* frame)
    {
        frame->m_pose->m_is_registered_to_graph = true;

        c_FramePose* pose = frame->m_pose;

        m_all_frame_poses_mutex.lock();
        m_all_frame_poses.push_back(pose);
        m_all_frame_poses_mutex.unlock();
    }

    void c_KeyFrameGraph::insert_constraint(c_KFConstraintStruct* constraint)
    {
        c_edge_sim3* edge = new c_edge_sim3();
        edge->setId(m_next_edge_id);
        ++m_next_edge_id;

        edge->setMeasurement(constraint->m_second_to_first);
        edge->setInformation(constraint->m_information);
        edge->setRobustKernel(constraint->m_robustKernel);

        edge->resize(2);
        
        assert(constraint->m_first_frame->m_pose->m_vertex != nullptr);
        edge->setVertex(0,constraint->m_first_frame->m_pose->m_vertex);

        assert(constraint->m_second_frame->m_pose->m_vertex != nullptr);
        edge->setVertex(1,constraint->m_second_frame->m_pose->m_vertex);

        constraint->m_edge = edge;

        m_new_edge_buffer.push_back(constraint);

        constraint->m_first_frame->m_neighbors.insert(constraint->m_second_frame);
        constraint->m_second_frame->m_neighbors.insert(constraint->m_first_frame);

        m_edges_lists_mutex.lock();
        constraint->m_index_in_all_edges = m_edges_all.size();
        m_edges_all.push_back(constraint);
        m_edges_lists_mutex.unlock();
    }

    int c_KeyFrameGraph::optimize(int num_iterations)
    {
	    //std::cout<<"g2o edges:"<<m_graph.edges().size()<<std::endl;
        if(m_graph.edges().size() == 0)
        {
            return 0;
        }

        m_graph.setVerbose(false);
	    //m_graph.setVerbose(true);
        m_graph.initializeOptimization();

        return m_graph.optimize(num_iterations,false);
    }

    bool c_KeyFrameGraph::add_elements_from_buffer()
    {
        bool is_added = false;

        m_keyframes_for_retrack_mutex.lock();
        for(int i = 0;i < m_new_keyframes_buffer.size();i++)
        {
            lsd_slam::c_Frame* new_kf = m_new_keyframes_buffer[i];
            m_graph.addVertex(new_kf->m_pose->m_vertex);
            assert(!new_kf->m_pose->m_is_in_graph);
            new_kf->m_pose->m_is_in_graph = true;

            m_keyframes_for_retrack.push_back(new_kf);

            is_added = true;
        }
        m_keyframes_for_retrack_mutex.unlock();

        m_new_keyframes_buffer.clear();

        for(int i = 0;i < m_new_edge_buffer.size();i++)
        {
            lsd_slam::c_KFConstraintStruct* edge = m_new_edge_buffer[i];
            m_graph.addEdge(edge->m_edge);
            is_added = true;
        }

        m_new_edge_buffer.clear();

        return is_added;
    }

    void c_KeyFrameGraph::calculate_graph_distances_to_frame(c_Frame* start_frame,std::unordered_map<c_Frame*,int>* distance_map)
    {
        distance_map->insert(std::make_pair(start_frame,0));

        std::multimap<int,c_Frame*> priority_queue;

        priority_queue.insert(std::make_pair(0,start_frame));

        while(!priority_queue.empty())
        {
            auto it = priority_queue.begin();

            int length = it->first;
            c_Frame* frame = it->second;
            priority_queue.erase(it);

            auto map_entry = distance_map->find(frame);

            if(map_entry != distance_map->end() && length > map_entry->second)
            {
                continue;
            }
            
            for(c_Frame* neighbor : frame->m_neighbors)
            {
                auto neighbor_map_entry = distance_map->find(neighbor);

                if(neighbor_map_entry != distance_map->end() && length + 1 >= neighbor_map_entry->second)
                {
                    continue;
                }

                if(neighbor_map_entry != distance_map->end())
                {
                    neighbor_map_entry->second = length+1;
                }
                else
                {
                    distance_map->insert(std::make_pair(neighbor,length+1));
                }

                priority_queue.insert(std::make_pair(length+1,neighbor));
            }
        }
    }

    void c_KeyFrameGraph::find_euclidean_overlap_frames(
        c_Frame* frame,float distance_threshold,float angle_threshold,
        bool check_both_scales,
        std::vector<c_TrackableKFStruct,Eigen::aligned_allocator<c_TrackableKFStruct> >& potential_reference_frames
    )
    {
        float cos_angle_threshold = cosf(angle_threshold * 0.5f * (m_fowx+m_fowy));

        Eigen::Vector3d pos = frame->get_scaled_cam_to_world().translation();
        Eigen::Vector3d viewing_direction = frame->get_scaled_cam_to_world().rotationMatrix().rightCols<1>();

        float dist_fac_reciprocal = 1;
        if(check_both_scales)
        {
            dist_fac_reciprocal = frame->m_mean_idepth / frame->get_scaled_cam_to_world().scale();
        }

        m_keyframes_all_mutex.lock_shared();
        for(int i = 0;i < m_keyframes_all.size();i++)
        {
            c_Frame* other = m_keyframes_all[i];

            Eigen::Vector3d other_pos = other->get_scaled_cam_to_world().translation();
            
            float dist_fac = other->m_mean_idepth / other->get_scaled_cam_to_world().scale();
            if(check_both_scales && dist_fac_reciprocal < dist_fac)
            {
                dist_fac = dist_fac_reciprocal;
            }
            Eigen::Vector3d dist = (pos - other_pos) * dist_fac;
            float dist_norm2 = dist.dot(dist);
            if(dist_norm2 > distance_threshold)
                continue;
            
            Eigen::Vector3d other_viewing_direction = other->get_scaled_cam_to_world().rotationMatrix().rightCols<1>();
            float direction_dot_prod = other_viewing_direction.dot(viewing_direction);//
            if(direction_dot_prod < cos_angle_threshold)
                continue;
            
            potential_reference_frames.push_back(c_TrackableKFStruct());
            potential_reference_frames.back().m_reference = other;
            potential_reference_frames.back().m_reference_to_frame = se3FromSim3(other->get_scaled_cam_to_world().inverse() * frame->get_scaled_cam_to_world()).inverse();
            potential_reference_frames.back().m_distance = dist_norm2;
            potential_reference_frames.back().m_angle = direction_dot_prod;

        }

        m_keyframes_all_mutex.unlock_shared();
    }

    void c_KeyFrameGraph::find_candidates(
        c_Frame* keyframe,float closeness_threshold,
        std::unordered_set<c_Frame*,std::hash<c_Frame*>,std::equal_to<c_Frame*>,Eigen::aligned_allocator<c_Frame*> >& results
    )
    {
        std::vector<c_TrackableKFStruct,Eigen::aligned_allocator<c_TrackableKFStruct> > potential_reference_frames;

        float distance_threshold = closeness_threshold * 15 / (KFDistWeight*KFDistWeight);
        float angle_threshold = 1.0 - 0.25 * closeness_threshold;
        bool check_both_scales = true;
        find_euclidean_overlap_frames(keyframe,distance_threshold,angle_threshold,check_both_scales,potential_reference_frames);

        for(int i = 0;i < potential_reference_frames.size();i++)
            results.insert(potential_reference_frames[i].m_reference);
    }

    c_Frame* c_KeyFrameGraph::find_reposition_candidate(c_Frame* frame,float max_score)
    {
        std::vector<c_TrackableKFStruct,Eigen::aligned_allocator<c_TrackableKFStruct> > potential_reference_frames;
        float distance_threshold = max_score / (KFDistWeight*KFDistWeight);
        float angle_threshold = 0.75;
        bool check_both_scales = false;
        find_euclidean_overlap_frames(frame,distance_threshold,angle_threshold,check_both_scales,potential_reference_frames);

        float best_score = max_score;
        float best_dist,best_usage;
        float best_pose_discrepancy = 0;
        c_Frame* best_frame = 0;
        SE3 best_reference_to_frame = SE3();
        SE3 best_reference_to_frame_tracked = SE3();

        int checked_secondary = 0;
        for(int i = 0;i < potential_reference_frames.size();i++)
        {
            c_TrackableKFStruct* ref_struct = &(potential_reference_frames[i]);

            if(frame->get_tracking_parent() == ref_struct->m_reference)
            {
                continue;
            }
            if(ref_struct->m_reference->m_index_in_keyframes < INITIALIZATION_PHASE_COUNT)
            {
                continue;
            }

            m_se3_tracker->check_overlap(ref_struct->m_reference,ref_struct->m_reference_to_frame);

            float score = get_reference_frame_score(ref_struct->m_distance,m_se3_tracker->m_point_usage);

            if(score < max_score)
            {
                SE3 reference_to_frame_tracked = m_se3_tracker->quick_trackFrame(ref_struct->m_reference,frame,ref_struct->m_reference_to_frame);
                Eigen::Vector3d dist = reference_to_frame_tracked.translation() * ref_struct->m_reference->m_mean_idepth;

                float new_score = get_reference_frame_score(dist.dot(dist),m_se3_tracker->m_point_usage);
                float pose_discrepancy = (ref_struct->m_reference_to_frame * reference_to_frame_tracked.inverse()).log().norm();

                float good_var = m_se3_tracker->m_point_usage * m_se3_tracker->m_last_good_count / (m_se3_tracker->m_last_good_count + m_se3_tracker->m_last_bad_count);

                checked_secondary++;

                if(m_se3_tracker->m_tracking_was_good && good_var > relocalizationTH 
                    && new_score < best_score && pose_discrepancy < 0.2)
                {
                    best_pose_discrepancy = pose_discrepancy;
                    best_score = new_score;
                    best_frame = ref_struct->m_reference;
                    best_reference_to_frame = ref_struct->m_reference_to_frame;
                    best_reference_to_frame_tracked = reference_to_frame_tracked;
                    best_dist = dist.dot(dist);
                    best_usage = m_se3_tracker->m_point_usage;
                }

            }

        }

        if(best_frame != 0)
            return best_frame;
        return 0;
    }

    void c_KeyFrameGraph::merge_optimization_result()
    {
        m_pose_consistency_mutex.lock();

        if(m_have_unmerged_optimization)
        {
            m_keyframes_all_mutex.lock_shared();
            for (int i = 0; i < m_keyframes_all.size(); i++)
            {
                lsd_slam::c_Frame* keyframe = m_keyframes_all[i];
                keyframe->m_pose->apply_pose_graph_opt_result();
            }
            m_keyframes_all_mutex.unlock_shared();

            m_have_unmerged_optimization = false;
        }

        m_pose_consistency_mutex.unlock();
    }

    void c_KeyFrameGraph::print_all_pose()
    {
        std::string filename = "D:\\slam_all\\init\\my_pose.txt";
        FILE* fp = fopen(filename.c_str(), "w+");
        //int n = m_keyframes_all.size();
        int n = m_all_frame_poses.size();
        for (int i = 0; i < n; i++)
        {
            //SE3 pose = se3FromSim3(m_keyframes_all[i]->m_pose->get_cam_to_world());
            SE3 pose = se3FromSim3(m_all_frame_poses[i]->get_cam_to_world());
            Eigen::Vector3f trans = pose.translation().cast<float>();
            Eigen::Quaternionf rot = pose.unit_quaternion().cast<float>();

            fprintf(fp, "%d ", m_all_frame_poses[i]->m_frame_ID);
            for (int j = 0; j < 3; j++)
                fprintf(fp, "%f ", trans[j]);
            fprintf(fp, "%f %f %f %f", rot.x(), rot.y(), rot.z(), rot.w());

            fprintf(fp, "\n");
        }
        fclose(fp);
    }

    //fabmap
    c_Frame* c_KeyFrameGraph::find_appearance_based_candidate(c_Frame* keyframe)
    {
        if (!m_fabmap.is_valid())
        {
            printf("error call fabmap!\n");
            return nullptr;
        }

        int new_id, loop_id;
        cv::Mat keyFrameImage(keyframe->get_height(), keyframe->get_width(), CV_32F, const_cast<float*>(keyframe->get_image()));
        m_fabmap.compareAndAdd(keyFrameImage, &new_id, &loop_id);
        if (new_id < 0)
            return nullptr;
        m_fabmap_id_to_KF.insert(std::make_pair(new_id, keyframe));
        if (loop_id >= 0)
            return m_fabmap_id_to_KF.at(loop_id);
        else
            return nullptr;
    }
}
