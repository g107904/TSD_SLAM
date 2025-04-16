#include "c_MapOptimization.h"

#include "DataStructures/c_Frame.h"
#include "DataStructures/c_FrameMemory.h"

#include "Tracking/c_SE3Tracker.h"
#include "Tracking/c_Sim3Tracker.h"
#include "Tracking/c_TrackingReference.h"

#include "GlobalMapping/c_KeyFrameGraph.h"
#include "GlobalMapping/g2o_with_type_sim3.h"

#include "util/interpolation.h"

#include <g2o/core/robust_kernel_impl.h>
#include <deque>


namespace lsd_slam
{	

    c_MapOptimization::c_MapOptimization(int w,int h,Eigen::Matrix3f K)
    {
        m_width = w;
        m_height = h;
        m_K = K;

        m_keyframe_graph = new c_KeyFrameGraph(w,h,K);
        m_new_constraint_added = false;
       	
	m_failed_to_retrack= 0;	
	
        m_do_final_optimization = false;

        m_se3_constraint_tracker = new c_SE3Tracker(w,h,K);
        m_sim3_constraint_tracker = new c_Sim3Tracker(w,h,K);
        m_new_KF_tracking_reference = new c_TrackingReference();
        m_candidate_tracking_reference = new c_TrackingReference();

        //finalize args
        m_do_full_reconstraint_track = false;
        m_do_final_optimization = false;
        
    }

    c_MapOptimization::~c_MapOptimization()
    {
        delete m_keyframe_graph;
        delete m_se3_constraint_tracker;
        delete m_sim3_constraint_tracker;
        delete m_new_KF_tracking_reference;
        delete m_candidate_tracking_reference;
    }

    void c_MapOptimization::do_constraint_search(std::unique_lock<std::mutex>& lock)
    {
        //std::unique_lock<std::mutex> lock(m_keyframe_graph->m_new_keyframes_mutex);


        if(m_keyframe_graph->m_new_keyframes.size() == 0)
        {
            lock.unlock();
            m_keyframe_graph->m_keyframes_for_retrack_mutex.lock();
            bool has_done_something = false;

            if(m_keyframe_graph->m_keyframes_for_retrack.size() > 10)
            {
                std::deque<c_Frame*>::iterator random_to_retrack_frame = 
                    m_keyframe_graph->m_keyframes_for_retrack.begin() 
                    + (rand() % (m_keyframe_graph->m_keyframes_for_retrack.size()/3));
                
                c_Frame* frame = *random_to_retrack_frame;

                m_keyframe_graph->m_keyframes_for_retrack.erase(random_to_retrack_frame);
                m_keyframe_graph->m_keyframes_for_retrack.push_back(frame);
                m_keyframe_graph->m_keyframes_for_retrack_mutex.unlock();

                int found = find_constraints_for_new_KF(frame,false,2.0);

		        //std::cout<<"random:"<<found<<std::endl;
		        //std::cout<<"fail"<<m_failed_to_retrack<<' '<<(int)m_keyframe_graph->m_keyframes_for_retrack.size()-5<<std::endl;

                if(found == 0)
                    m_failed_to_retrack++;
                else
                    m_failed_to_retrack = 0;
                
                if(m_failed_to_retrack < (int) m_keyframe_graph->m_keyframes_for_retrack.size() - 5)
                    has_done_something = true;
            }
            else
                m_keyframe_graph->m_keyframes_for_retrack_mutex.unlock();

            lock.lock();
	
	    //std::cout<<"has done"<<has_done_something<<std::endl;
            
            if(!has_done_something)
	{
		
                m_keyframe_graph->m_new_keyframes_created_signal.wait_for(lock,std::chrono::milliseconds(500)); //1000
	}
        }

        else
        {
            c_Frame* new_KF = m_keyframe_graph->m_new_keyframes.front();

            m_keyframe_graph->m_new_keyframes.pop_front();
            lock.unlock();

            int found = find_constraints_for_new_KF(new_KF,true,1.0);
	
	    std::cout<<"new:"<<found<<std::endl;

            c_FrameMemory::get_instance().prune_active_frames();
            lock.lock();
        }

        if(m_do_full_reconstraint_track)
        {
            lock.unlock();
            int added = 0;
            for(int i = 0;i < m_keyframe_graph->m_keyframes_all.size();i++)
            {
                lsd_slam::c_Frame* keyframe = m_keyframe_graph->m_keyframes_all[i];
                if(keyframe->m_pose->m_is_in_graph)
                {
                    added += find_constraints_for_new_KF(
                        m_keyframe_graph->m_keyframes_all[i],false,1.0);
                }
            }

            m_do_full_reconstraint_track = false;

            m_last_num_constraints_added_on_full_retrack = added;
            lock.lock();
            
        }
    }

    int c_MapOptimization::find_constraints_for_new_KF(
        c_Frame* frame,bool force_parent,
        float close_candidates_threshold )
    {
        if(!frame->has_tracking_parent())
        {
            m_new_constraint_mutex.lock();
            m_keyframe_graph->add_keyframe(frame);
            m_new_constraint_added = true;
            m_new_constraint_created_signal.notify_all();
            m_new_constraint_mutex.unlock();
            return 0;
        }

        if(!force_parent 
            && (frame->m_last_constraint_tracked_cam_to_world 
            * frame->get_scaled_cam_to_world().inverse()).log().norm() < 0.01)
        {
            return 0;
        }

        frame->m_last_constraint_tracked_cam_to_world = frame->get_scaled_cam_to_world();

        //results
        //std::vector<c_KFConstraintStruct*,Eigen::aligned_allocator<c_KFConstraintStruct*> > constraints;
        std::vector<c_KFConstraintStruct* > constraints;


         std::unordered_set<c_Frame*,std::hash<c_Frame*>,std::equal_to<c_Frame*>,
            Eigen::aligned_allocator<c_Frame*> > candidates;
        
        m_keyframe_graph->find_candidates(frame,close_candidates_threshold,candidates);

        std::map<c_Frame*,Sim3,std::less<c_Frame*>,
            Eigen::aligned_allocator<std::pair<c_Frame*,Sim3> > > map_candidates_to_frame_init;

        //erase these exist neighbors
        for(auto it = candidates.begin();it != candidates.end();)
        {
            if(frame->m_neighbors.find(*it) != frame->m_neighbors.end())
            {
                it = candidates.erase(it);
            }
            else
                ++it;
        }

        m_keyframe_graph->m_pose_consistency_mutex.lock_shared();
        //init candidates sim3 map
        for(c_Frame* candidate : candidates)
        {
            Sim3 candidate_to_frame_init = frame->get_scaled_cam_to_world().inverse() * candidate->get_scaled_cam_to_world();
            map_candidates_to_frame_init[candidate] = candidate_to_frame_init;
        }
        //get distances to frame
        std::unordered_map<c_Frame*,int> distances_to_frame;
        if(frame->has_tracking_parent())
        {
            m_keyframe_graph->calculate_graph_distances_to_frame(frame->get_tracking_parent(),&distances_to_frame);
        }
        m_keyframe_graph->m_pose_consistency_mutex.unlock_shared();

        //close and far candidates
        std::unordered_set<c_Frame*,std::hash<c_Frame*>,std::equal_to<c_Frame*>,
            Eigen::aligned_allocator<c_Frame*> > close_candidates;
        //std::vector<c_Frame*,Eigen::aligned_allocator<c_Frame*> > far_candidates;
        std::vector<c_Frame*> far_candidates;

        c_Frame* parent = frame->has_tracking_parent() ? frame->get_tracking_parent() : 0;

        int close_failed = 0;
        int close_inconsistent = 0;

        SO3 disturbance = SO3::exp(Sophus::Vector3d(0.05,0,0));

        for(c_Frame* candidate : candidates)
        {
            if(candidate->get_id() == frame->get_id())
                continue;
            if(!candidate->m_pose->m_is_in_graph)
                continue;
            if(frame->has_tracking_parent() && candidate == frame->get_tracking_parent())
                continue;
            if(candidate->m_index_in_keyframes < INITIALIZATION_PHASE_COUNT)
                continue;

            SE3 candidate_to_frame_init = se3FromSim3(
                map_candidates_to_frame_init[candidate].inverse()).inverse();
            candidate_to_frame_init.so3() = candidate_to_frame_init.so3() * disturbance;

            SE3 candidate_to_frame = m_se3_constraint_tracker->quick_trackFrame(
                candidate,frame,candidate_to_frame_init);
            if(!m_se3_constraint_tracker->m_tracking_was_good)
            {
                close_failed++;
                continue;
            }

            SE3 frame_to_candidate_init = se3FromSim3(
                map_candidates_to_frame_init[candidate]).inverse();
            frame_to_candidate_init.so3() = disturbance * frame_to_candidate_init.so3();

            SE3 frame_to_candidate = m_se3_constraint_tracker->quick_trackFrame(
                frame,candidate,frame_to_candidate_init);
            if(!m_se3_constraint_tracker->m_tracking_was_good)
            {
                close_failed++;
                continue;
            }

            if((frame_to_candidate.so3() * candidate_to_frame.so3()).log().norm() >= 0.09)
            {
                close_inconsistent++;
                continue;
            }

            close_candidates.insert(candidate);
        }

        for(c_Frame* candidate : candidates)
        {
            if(candidate->get_id() == frame->get_id())
                continue;
            if(!candidate->m_pose->m_is_in_graph)
                continue;
            if(frame->has_tracking_parent() && candidate == frame->get_tracking_parent())
                continue;
            if(candidate->m_index_in_keyframes < INITIALIZATION_PHASE_COUNT)
                continue;

            if(distances_to_frame.at(candidate) < 4)
                continue;
            
            far_candidates.push_back(candidate);
        }

        int num_close = close_candidates.size();
        int num_far = far_candidates.size();

        //erase who has already tried (in close)
        for(auto it = close_candidates.begin();it != close_candidates.end();)
        {
            if(frame->m_tracking_failed.find(*it) == frame->m_tracking_failed.end())
            {
                ++it;
                continue;
            }
            auto range = frame->m_tracking_failed.equal_range(*it);
            bool skip = false;
            Sim3 frame_to_candidate = map_candidates_to_frame_init[*it].inverse();
            for(auto range_it = range.first;range_it != range.second;++range_it)
            {
                if((frame_to_candidate * range_it->second).log().norm() < 0.1)
                {
                    skip = true;
                    break;
                }
            }

            if(skip)
            {
                it = close_candidates.erase(it);
            }
            else
                ++it;
        }

        //erase who are already neighbors(in far)
        for(int i = 0;i < far_candidates.size();i++)
        {
            if(frame->m_tracking_failed.find(far_candidates[i]) == frame->m_tracking_failed.end())
                continue;
            
            auto range = frame->m_tracking_failed.equal_range(far_candidates[i]);

            bool skip = false;
            for(auto range_it = range.first;range_it != range.second;++range_it)
            {
                if((range_it->second).log().norm() < 0.2)
                {
                    skip = true;
                    break;
                }
            }
            if(skip)
            {
                far_candidates[i] = far_candidates.back();
                far_candidates.pop_back();
                i--;
            }
        }

        //reduce close number
        while((int)close_candidates.size() > maxLoopClosureCandidates)
        {
            c_Frame* worst = 0;
            int worst_neighbors = 0;
            for(c_Frame* candidate : close_candidates)
            {
                int neighbors_in_close = 0;
                for(c_Frame* neighbor : candidate->m_neighbors)
                {
                    if(close_candidates.find(neighbor) != close_candidates.end())
                        neighbors_in_close++;
                }
                if(neighbors_in_close > worst_neighbors || worst == 0)
                {
                    worst = candidate;
                    worst_neighbors = neighbors_in_close;
                }  
            }
            close_candidates.erase(worst);
        }

	//std::vector<c_Frame*,Eigen::aligned_allocator<c_Frame*> > tmp_close_candidates;
    std::vector<c_Frame*> tmp_close_candidates;
	for(auto candidate : close_candidates)
		tmp_close_candidates.push_back(candidate);	
	for(int i = 0;i < tmp_close_candidates.size();i++)
		for(int j = i+1;j < tmp_close_candidates.size();j++)
		{
            lsd_slam::c_Frame* frame_j = tmp_close_candidates[j];
            lsd_slam::c_Frame* frame_i = tmp_close_candidates[i];
		    if(frame_j->get_id() < frame_i->get_id())
		        {c_Frame* t = tmp_close_candidates[i];tmp_close_candidates[i] = tmp_close_candidates[j];tmp_close_candidates[j] = t;}
	    }
        
        //reduce far number
        int max_num_far_candidates = (maxLoopClosureCandidates+1)/2;
        if(max_num_far_candidates < 5)
            max_num_far_candidates = 5;
        while((int) far_candidates.size() > max_num_far_candidates)
        {
            int to_delete = rand() % far_candidates.size();
            far_candidates[to_delete] = far_candidates.back();
            far_candidates.pop_back();
        }

        //--------------start tracking----------------------
        m_new_KF_tracking_reference->import_frame(frame);

        for(c_Frame* candidate : tmp_close_candidates)
        {
            c_KFConstraintStruct* edge_1 = 0;
            c_KFConstraintStruct* edge_2 = 0;
            test_constraint(
                candidate,
                edge_1,edge_2,
                map_candidates_to_frame_init[candidate],
                loopclosureStrictness
            );

            if(edge_1 != 0)
            {
                constraints.push_back(edge_1);
                constraints.push_back(edge_2);

                //erase candidate from far
                for(int i = 0;i < far_candidates.size();i++)
                {
                    if(far_candidates[i] == candidate)
                    {
                        far_candidates[i] = far_candidates.back();
                        far_candidates.pop_back();
                    }
                }
            }
        }

        for(c_Frame* candidate : far_candidates)
        {
            c_KFConstraintStruct* edge_1 = 0;
            c_KFConstraintStruct* edge_2 = 0;
            test_constraint(
                candidate,
                edge_1,edge_2,
                Sim3(),
                loopclosureStrictness
            );
            
            if(edge_1 != 0)
            {
                constraints.push_back(edge_1);
                constraints.push_back(edge_2);
            }
        }

        if(parent != 0 && force_parent)
        {
            c_KFConstraintStruct* edge_1 = 0;
            c_KFConstraintStruct* edge_2 = 0;
            test_constraint(
                parent,
                edge_1,edge_2,
                map_candidates_to_frame_init[parent],
                100
            );

            if(edge_1 != 0)
            {
                constraints.push_back(edge_1);
                constraints.push_back(edge_2);
            }
            else //?hack
            {
                float down_weight_factor = 5;
                const float kernel_delta = 5 * sqrt(6000 * loopclosureStrictness) / down_weight_factor;

                m_keyframe_graph->m_pose_consistency_mutex.lock_shared();
                constraints.push_back(new c_KFConstraintStruct());
                lsd_slam::c_KFConstraintStruct* constraint = constraints.back();
                constraint->m_first_frame = frame;
                constraint->m_second_frame = parent;
                constraint->m_second_to_first = frame->get_scaled_cam_to_world().inverse() * parent->get_scaled_cam_to_world();
                constraint->m_information  <<
					0.8098,-0.1507,-0.0557, 0.1211, 0.7657, 0.0120, 0,
					-0.1507, 2.1724,-0.1103,-1.9279,-0.1182, 0.1943, 0,
					-0.0557,-0.1103, 0.2643,-0.0021,-0.0657,-0.0028, 0.0304,
					 0.1211,-1.9279,-0.0021, 2.3110, 0.1039,-0.0934, 0.0005,
					 0.7657,-0.1182,-0.0657, 0.1039, 1.0545, 0.0743,-0.0028,
					 0.0120, 0.1943,-0.0028,-0.0934, 0.0743, 0.4511, 0,
					0,0, 0.0304, 0.0005,-0.0028, 0, 0.0228;
                constraint->m_information *= (1e9/(down_weight_factor*down_weight_factor));
                constraint->m_robustKernel = new g2o::RobustKernelHuber();
                constraint->m_robustKernel->setDelta(kernel_delta);
                constraint->m_mean_residual = 10;
                constraint->m_mean_photometric_residual = 10;
                constraint->m_mean_depth_residual = 10;
                constraint->m_usage = 0;

                m_keyframe_graph->m_pose_consistency_mutex.unlock_shared();
            }
        }

        m_new_constraint_mutex.lock();
        m_keyframe_graph->add_keyframe(frame);
        for(int i = 0;i < constraints.size();i++)
            m_keyframe_graph->insert_constraint(constraints[i]);

	//for(int i = 0;i < constraints.size();i+=2) std::cout<<"edge:"<<constraints[i]->m_first_frame->get_id()<<' '<<constraints[i]->m_second_frame->get_id()<<std::endl;
        

        m_new_constraint_added = true;
        m_new_constraint_created_signal.notify_all();
        m_new_constraint_mutex.unlock();

        m_new_KF_tracking_reference->invalidate();
        m_candidate_tracking_reference->invalidate();

        return constraints.size();

    }

    void c_MapOptimization::test_constraint(
        c_Frame* candidate,
        c_KFConstraintStruct* &edge_1,c_KFConstraintStruct* &edge_2,
        Sim3& candidate_to_frame_init,
        float strictness)
    {
        m_candidate_tracking_reference->invalidate();

        m_candidate_tracking_reference->import_frame(candidate);
        Sim3 frame_to_candidate = candidate_to_frame_init.inverse();
        Sim3 candidate_to_frame = candidate_to_frame_init;

        Eigen::Matrix<float,7,7> frame_to_candidate_information,candidate_to_frame_information;

        //level 3
        float error_level3 = try_track_sim3(
            m_new_KF_tracking_reference,m_candidate_tracking_reference,
            SIM3TRACKING_MAX_LEVEL-1,3,
            frame_to_candidate,candidate_to_frame
        );

        if(error_level3 > 3000 * strictness)
        {
            edge_1 = edge_2 = 0;
            m_new_KF_tracking_reference->m_keyframe->m_tracking_failed.insert(
                std::pair<c_Frame*,Sim3>(candidate,candidate_to_frame_init));
            return;
        }

        //level 2
        float error_level2 = try_track_sim3(
            m_new_KF_tracking_reference,m_candidate_tracking_reference,
            2,2,
            frame_to_candidate,candidate_to_frame
        );

        if(error_level2 > 4000 * strictness)
        {
            edge_1 = edge_2;
             m_new_KF_tracking_reference->m_keyframe->m_tracking_failed.insert(
                std::pair<c_Frame*,Sim3>(candidate,candidate_to_frame_init));
            return;
        }

        edge_1 = new c_KFConstraintStruct();
        edge_2 = new c_KFConstraintStruct();

        //level 1
        float error_level1 = try_track_sim3(
            m_new_KF_tracking_reference,m_candidate_tracking_reference,
            1,1,
            frame_to_candidate,candidate_to_frame,
            edge_1,edge_2
        );

        if(error_level1 > 6000*strictness)
        {
            delete edge_1;
            delete edge_2;
            edge_1 = edge_2 = 0;
            m_new_KF_tracking_reference->m_keyframe->m_tracking_failed.insert(
                std::pair<c_Frame*,Sim3>(candidate,candidate_to_frame_init));
            return;
        }

        const float kernel_delta = 5 * sqrt(6000 * loopclosureStrictness);
        edge_1->m_robustKernel = new g2o::RobustKernelHuber();
        edge_1->m_robustKernel->setDelta(kernel_delta);
        edge_2->m_robustKernel = new g2o::RobustKernelHuber();
        edge_2->m_robustKernel->setDelta(kernel_delta);
    }

    float c_MapOptimization::try_track_sim3(
        c_TrackingReference* A,c_TrackingReference* B,
        int start_level,int end_level,
        Sim3& A_to_B,Sim3& B_to_A,
        c_KFConstraintStruct* edge_1,c_KFConstraintStruct* edge_2
        )
    {
        B_to_A  = m_sim3_constraint_tracker->trackFrame(
            A,B->m_keyframe,
            B_to_A,
            start_level,end_level
        );
        Eigen::Matrix<float,7,7> B_to_A_information = m_sim3_constraint_tracker->m_last_sim3_hessian;
        float B_to_A_mean_residual = m_sim3_constraint_tracker->m_last_residual;
        float B_to_A_mean_depth_residual = m_sim3_constraint_tracker->m_last_depth_residual;
        float B_to_A_mean_photometric_residual = m_sim3_constraint_tracker->m_last_photometric_residual;
        float B_to_A_usage = m_sim3_constraint_tracker->m_point_usage;

        if(
            m_sim3_constraint_tracker->m_diverged 
            || B_to_A.scale() > 1 / Sophus::SophusConstants<sophusType>::epsilon()
            || B_to_A.scale() < Sophus::SophusConstants<sophusType>::epsilon()
            || B_to_A_information(0,0) == 0 
            || B_to_A_information(6,6) == 0)
        {
            return 1e20;
        }

        A_to_B = m_sim3_constraint_tracker->trackFrame(
            B,A->m_keyframe,
            A_to_B,
            start_level,end_level
        );
        Eigen::Matrix<float,7,7> A_to_B_information = m_sim3_constraint_tracker->m_last_sim3_hessian;
        float A_to_B_mean_residual = m_sim3_constraint_tracker->m_last_residual;
        float A_to_B_mean_depth_residual = m_sim3_constraint_tracker->m_last_depth_residual;
        float A_to_B_mean_photometric_residual = m_sim3_constraint_tracker->m_last_photometric_residual;
        float A_to_B_usage = m_sim3_constraint_tracker->m_point_usage;
        
        if(
            m_sim3_constraint_tracker->m_diverged 
            || A_to_B.scale() > 1 / Sophus::SophusConstants<sophusType>::epsilon()
            || A_to_B.scale() < Sophus::SophusConstants<sophusType>::epsilon()
            || A_to_B_information(0,0) == 0 
            || A_to_B_information(6,6) == 0)
        {
            return 1e20;
        }

        // Propagate uncertainty (with d(a * b) / d(b) = Adj_a) and calculate Mahalanobis norm
        Eigen::Matrix<float,7,7> datimesb_db = A_to_B.cast<float>().Adj();
        Eigen::Matrix<float,7,7> diff_hesse = (A_to_B_information.inverse() + datimesb_db * B_to_A_information.inverse() * datimesb_db.transpose()).inverse();
        Eigen::Matrix<float,7,1> diff = (A_to_B * B_to_A).log().cast<float>();

        float reciprocal_consistency = (diff_hesse * diff).dot(diff);

        if(edge_1 != 0 && edge_2 != 0)
        {
            edge_1->m_first_frame = A->m_keyframe;
            edge_1->m_second_frame = B->m_keyframe;
            edge_1->m_second_to_first = B_to_A;
            edge_1->m_information = B_to_A_information.cast<double>();
            edge_1->m_mean_residual = B_to_A_mean_residual;
            edge_1->m_mean_depth_residual = B_to_A_mean_depth_residual;
            edge_1->m_mean_photometric_residual = B_to_A_mean_photometric_residual;
            edge_1->m_usage = B_to_A_usage;

            edge_2->m_first_frame = B->m_keyframe;
            edge_2->m_second_frame = A->m_keyframe;
            edge_2->m_second_to_first = A_to_B;
            edge_2->m_information = A_to_B_information.cast<double>();
            edge_2->m_mean_residual = A_to_B_mean_residual;
            edge_2->m_mean_depth_residual = A_to_B_mean_depth_residual;
            edge_2->m_mean_photometric_residual = A_to_B_mean_photometric_residual;
            edge_2->m_usage = A_to_B_usage;

            edge_1->m_reciprocal_consistency = edge_2->m_reciprocal_consistency = reciprocal_consistency;
        }

        return reciprocal_consistency;
    }

    bool c_MapOptimization::do_optimization_it(int iteration_per_try,float min_change)
    {
        m_g2o_graph_access_mutex.lock();

        m_new_constraint_mutex.lock();
        m_keyframe_graph->add_elements_from_buffer();
        m_new_constraint_mutex.unlock();

        int iterations = m_keyframe_graph->optimize(iteration_per_try);

        m_keyframe_graph->m_pose_consistency_mutex.lock_shared();
        m_keyframe_graph->m_keyframes_all_mutex.lock_shared();
        float max_change = 0;
        float sum_change = 0;
        float sum = 0;
        for(int i = 0;i < m_keyframe_graph->m_keyframes_all.size();i++)
        {
            lsd_slam::c_Frame* keyframe = m_keyframe_graph->m_keyframes_all[i];
            keyframe->m_edge_error_sum = 0;
            keyframe->m_edges_num = 0;

            if(!keyframe->m_pose->m_is_in_graph) 
                continue;

            Sim3 a = keyframe->m_pose->m_vertex->estimate();
            Sim3 b = keyframe->get_scaled_cam_to_world();
            Sophus::Vector7f diff = (a*b.inverse()).log().cast<float>();

            for(int j = 0;j < 7;j++)
            {
                float d = fabsf((float)(diff[j]));
                if(d > max_change)
                    max_change = d;
                sum_change += d;
            }
            sum += 7;

            keyframe->m_pose->set_pose_graph_opt_result(
                a
            );

            for(auto edge : keyframe->m_pose->m_vertex->edges())
            {
                keyframe->m_edge_error_sum += ((c_edge_sim3*)(edge))->chi2();
                keyframe->m_edges_num++;
            }
        }

        m_keyframe_graph->m_have_unmerged_optimization = true;
        m_keyframe_graph->m_keyframes_all_mutex.unlock_shared();
        m_keyframe_graph->m_pose_consistency_mutex.unlock_shared();

        m_g2o_graph_access_mutex.unlock();
/*
	printf("did %d optimization iterations. Max Pose Parameter Change: %f; avgChange: %f. %s\n", iterations, max_change, sum_change / sum,
			max_change > min_change && iterations == iteration_per_try ? "continue optimizing":"Waiting for addition to graph.");
*/
        return max_change > min_change && iterations == iteration_per_try;
        }
    
    void c_MapOptimization::do_optimization()
    {
        std::unique_lock<std::mutex> lock(m_new_constraint_mutex);
        if(!m_new_constraint_added)
        {
		m_new_constraint_created_signal.wait_for(lock,std::chrono::milliseconds(2000));//4000
		
		//debug
		//m_new_constraint_created_signal.wait(lock);
	}
        m_new_constraint_added = false;
        lock.unlock();

        if(m_do_final_optimization)
        {
            do_optimization_it(50,0.001);
            m_do_final_optimization = false;
        }

        while(do_optimization_it(5,0.02));
    }
}

