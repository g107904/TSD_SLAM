#include "Tracking/c_SE3Tracker.h"
#include "Tracking/c_TrackingReference.h"

#include "DepthEstimation/c_DepthMap.h"
#include "DepthEstimation/c_DepthMapEstimation.h"

#include "GlobalMapping/c_KeyFrameGraph.h"

#include "DataStructures/c_FrameMemory.h"
#include "DataStructures/c_Frame.h"


namespace lsd_slam
{
    c_DepthMapEstimation::c_DepthMapEstimation(int w,int h,Eigen::Matrix3f K,c_KeyFrameGraph* graph)
    {
        m_width = w;
        m_height = h;
        m_K = K;
        m_keyframe_graph = graph;

        m_tracking_is_good = true;

        m_current_keyframe = nullptr;
        m_tracking_reference_shared_PT = nullptr;
    
        m_should_create_new_KF = false;
        m_map = new c_DepthMap(w,h,K);
    
        m_tracking_reference = new c_TrackingReference();
        m_mapping_reference = new c_TrackingReference();
        m_relocalizer = new c_Relocalizer(w,h,K);
        m_se3_tracker = new c_SE3Tracker(w, h, K);

    }

    c_DepthMapEstimation::~c_DepthMapEstimation()
    {
        delete m_mapping_reference;

        delete m_map;

        delete m_tracking_reference;

        delete m_relocalizer;
    

        m_last_tracked_frame.reset();


        m_current_keyframe.reset();

        m_tracking_reference_shared_PT.reset();
    }

    void c_DepthMapEstimation::random_init(unsigned char* image,int id)
    {
        m_current_keyframe_mutex.lock();

        m_current_keyframe.reset(new c_Frame(id,m_width,m_height,m_K,image));

        m_map->initializeRandomly(m_current_keyframe.get());

        m_keyframe_graph->add_frame(m_current_keyframe.get());

        m_current_keyframe_mutex.unlock();

        m_keyframe_graph->m_id_to_keyframe_mutex.lock();
        m_keyframe_graph->m_id_to_keyframe.insert(
            std::make_pair(m_current_keyframe->get_id(),m_current_keyframe));
        m_keyframe_graph->m_id_to_keyframe_mutex.unlock();
    }

    void c_DepthMapEstimation::get_depth_init(unsigned char* image, float* depth,int id)
    {
        m_current_keyframe_mutex.lock();

        m_current_keyframe.reset(new c_Frame(id,m_width,m_height,m_K,image));
        //m_current_keyframe->set_idepth_and_var_from_ground_truth(depth);

        m_map->initializeFromGroundTruth(m_current_keyframe.get(),depth);

        m_keyframe_graph->add_frame(m_current_keyframe.get());

        m_current_keyframe_mutex.unlock();

        m_keyframe_graph->m_id_to_keyframe_mutex.lock();
        m_keyframe_graph->m_id_to_keyframe.insert(
            std::make_pair(m_current_keyframe->get_id(),m_current_keyframe));
        m_keyframe_graph->m_id_to_keyframe_mutex.unlock();
    }

    bool c_DepthMapEstimation::do_mapping()
    {
        if(m_current_keyframe == 0)
            return false;
        m_keyframe_graph->merge_optimization_result();

        if(m_tracking_is_good)
        {
            if(m_should_create_new_KF)
            {
                finish_current_KF();

                float max_score = 1.0f;
                c_Frame* new_reference_KF = m_keyframe_graph->find_reposition_candidate(m_last_tracked_frame.get(),max_score);
                if(new_reference_KF != 0)
                    load_new_current_KF_from_exist(new_reference_KF);
                else
                    create_new_current_KF(m_last_tracked_frame);
                m_should_create_new_KF = false;
            }
            else
            {
                bool did_something = update_KF();
                if(!did_something)
                    return false;
            }
            return true;
        }
        else
        {
            if(m_map->isValid())
            {
                if(m_current_keyframe->m_num_mapped_on_this_total >= MIN_NUM_MAPPED)
                    finish_current_KF();
                else
                    discard_current_KF();
                m_map->invalidate();
            }

            if(!m_relocalizer->m_is_thread_running)
                m_relocalizer->thread_start(m_keyframe_graph->m_keyframes_all);
            
            if(m_relocalizer->wait_relocalize_result(50))
                take_relocalize_result();
            
            return true;
        }
    }

    void c_DepthMapEstimation::finish_current_KF()
    {
        m_map->finalizeKeyFrame();

        m_mapping_reference->import_frame(m_current_keyframe.get());

        //todo
        m_current_keyframe->set_quick_data(m_mapping_reference);

        m_mapping_reference->invalidate();

        if(m_current_keyframe->m_index_in_keyframes < 0)
        {
            m_keyframe_graph->m_keyframes_all_mutex.lock();
            m_current_keyframe->m_index_in_keyframes = m_keyframe_graph->m_keyframes_all.size();
            m_keyframe_graph->m_keyframes_all.push_back(m_current_keyframe.get());
            m_keyframe_graph->m_keyframes_all_mutex.unlock();

            m_keyframe_graph->m_new_keyframes_mutex.lock();
            m_keyframe_graph->m_new_keyframes.push_back(m_current_keyframe.get());
            m_keyframe_graph->m_new_keyframes_created_signal.notify_all();
            m_keyframe_graph->m_new_keyframes_mutex.unlock();
        }

    }

    void c_DepthMapEstimation::discard_current_KF()
    {
        if(m_current_keyframe->m_index_in_keyframes >= 0)
        {
            finish_current_KF();
            return;
        }

        m_map->invalidate();

        m_keyframe_graph->m_all_frame_poses_mutex.lock();
        for(int i = 0;i < m_keyframe_graph->m_all_frame_poses.size();i++)
        {
            lsd_slam::c_FramePose* pose = m_keyframe_graph->m_all_frame_poses[i];
            if(pose->m_tracking_parent != 0 
            && pose->m_tracking_parent->m_frame_ID == m_current_keyframe->get_id())
            {
                pose->m_tracking_parent = 0;
            }
        }
        m_keyframe_graph->m_all_frame_poses_mutex.unlock();

        m_keyframe_graph->m_id_to_keyframe_mutex.lock();
        m_keyframe_graph->m_id_to_keyframe.erase(m_current_keyframe->get_id());
        m_keyframe_graph->m_id_to_keyframe_mutex.unlock();
    }

    void c_DepthMapEstimation::create_new_current_KF(std::shared_ptr<c_Frame> new_KF_candidate)
    {
        m_keyframe_graph->m_id_to_keyframe_mutex.lock();
        m_keyframe_graph->m_id_to_keyframe.insert(std::make_pair(new_KF_candidate->get_id(),new_KF_candidate));
        m_keyframe_graph->m_id_to_keyframe_mutex.unlock();

        m_map->createKeyframe(new_KF_candidate.get());

        m_current_keyframe_mutex.lock();
        m_current_keyframe = new_KF_candidate;
        m_current_keyframe_mutex.unlock();
    }

    void c_DepthMapEstimation::load_new_current_KF_from_exist(c_Frame* frame)
    {
        m_map->setFromExistingKF(frame);

        m_current_keyframe_mutex.lock();
        m_current_keyframe= m_keyframe_graph->m_id_to_keyframe.find(frame->get_id())->second;
        m_current_keyframe->m_has_depth_been_updated = false;
        m_current_keyframe_mutex.unlock();
    }

    bool c_DepthMapEstimation::update_KF()
    {
        std::shared_ptr<c_Frame> reference = nullptr;
        std::deque<std::shared_ptr<c_Frame> > references;

        m_unmapped_tracked_frames_mutex.lock();

        while(m_unmapped_tracked_frames.size() > 0
        &&      (!m_unmapped_tracked_frames.front()->has_tracking_parent()
                 || m_unmapped_tracked_frames.front()->get_tracking_parent() != m_current_keyframe.get() 
                )
            )
        {
            m_unmapped_tracked_frames.front()->clear_ref_pixel_was_good();
            m_unmapped_tracked_frames.pop_front();
        }

        if(m_unmapped_tracked_frames.size() > 0)
        {
            for(int i = 0;i < m_unmapped_tracked_frames.size();i++)
            {
                references.push_back(m_unmapped_tracked_frames[i]);
            }

            std::shared_ptr<c_Frame> popped = m_unmapped_tracked_frames.front();
            m_unmapped_tracked_frames.pop_front();
            m_unmapped_tracked_frames_mutex.unlock();

            m_map->updateKeyframe(references);
            popped->clear_ref_pixel_was_good();
            references.clear();
        }
        else
        {
            m_unmapped_tracked_frames_mutex.unlock();
            return false;
        }
        return true;
    }

    void c_DepthMapEstimation::take_relocalize_result()
    {
        c_Frame* keyframe;
        int success_frame_ID;
        SE3 success_to_KF_init;
        std::shared_ptr<c_Frame> success_frame;
        m_relocalizer->thread_stop();
        m_relocalizer->get_relocalize_result(keyframe,success_frame,success_frame_ID,success_to_KF_init);

        load_new_current_KF_from_exist(keyframe);

        m_current_keyframe_mutex.lock();
        m_tracking_reference->import_frame(m_current_keyframe.get());
        m_tracking_reference_shared_PT = m_current_keyframe;
        m_current_keyframe_mutex.unlock();

        m_se3_tracker->trackFrame(m_tracking_reference,success_frame.get(),success_to_KF_init);

        if(!m_se3_tracker->m_tracking_was_good || m_se3_tracker->m_last_good_count / (m_se3_tracker->m_last_good_count + m_se3_tracker->m_last_bad_count) < 1 - 0.75f * (1 - MIN_GOODPERALL_PIXEL))
        {
            m_tracking_reference->invalidate();
        }
        else
        {
            m_keyframe_graph->add_frame(success_frame.get());

            m_unmapped_tracked_frames_mutex.lock();
            if(m_unmapped_tracked_frames.size() < 50)
                m_unmapped_tracked_frames.push_back(success_frame);
            m_unmapped_tracked_frames_mutex.unlock();

            m_current_keyframe_mutex.lock();
            m_should_create_new_KF = false;
            m_tracking_is_good = true;
            m_current_keyframe_mutex.unlock();

        }
    }

    void c_DepthMapEstimation::set_current_KF(std::shared_ptr<c_Frame>& frame_pt)
    {
        m_current_keyframe_mutex.lock();
        m_current_keyframe = frame_pt;
        m_current_keyframe_mutex.unlock();
    }

    void c_DepthMapEstimation::set_tracking_is_good(bool flag)
    {
        m_tracking_is_good_mutex.lock();
        m_tracking_is_good = flag;
        m_tracking_is_good_mutex.unlock();
    }

    void c_DepthMapEstimation::set_last_tracked_frame(c_Frame* frame)
    {
        m_last_tracked_frame_mutex.lock();
        m_last_tracked_frame.reset(frame);
        m_last_tracked_frame_mutex.unlock();
    }


}
