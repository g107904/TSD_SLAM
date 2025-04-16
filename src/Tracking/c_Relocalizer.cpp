#include "c_Relocalizer.h"
#include "DataStructures/c_Frame.h"
#include "Tracking/c_SE3Tracker.h"


namespace lsd_slam
{
    c_Relocalizer::c_Relocalizer(int w,int h,Eigen::Matrix3f K)
    {
        m_width = w;
        m_height = h;
        m_K = K;

        m_KF_for_relocalize.clear();
        m_next_index = m_max_index = 0;
        m_is_thread_running = m_keep_running = false;

        m_has_result = false;
        m_result_KF = 0;
        m_result_frame_ID = 0;
        m_result_frame_to_keyframe = SE3();
    }

    c_Relocalizer::~c_Relocalizer()
    {
        if(m_is_thread_running)
            thread_stop();
    }

    void c_Relocalizer::update_current_frame(std::shared_ptr<c_Frame> current_frame)
    {
        std::unique_lock<std::mutex> lock(m_ex_mutex);

        if(m_has_result)
            return;
        this->m_current_relocalize_frame = current_frame;
        m_max_index = m_next_index + m_KF_for_relocalize.size();
        m_new_current_frame_signal.notify_all();
        lock.unlock();
    }

    void c_Relocalizer::thread_start(std::vector<c_Frame* >& keyframe_all)
    {
        m_KF_for_relocalize.clear();
        int len = keyframe_all.size();
        for(int i = 0;i < len;i++)
        {
            m_KF_for_relocalize.push_back(keyframe_all[i]);
            int rand_index = rand() % (m_KF_for_relocalize.size());

            m_KF_for_relocalize.back() = m_KF_for_relocalize[rand_index];
            m_KF_for_relocalize[rand_index] = m_KF_for_relocalize.back();
        }
        m_next_index = 0;
        m_max_index = m_KF_for_relocalize.size();

        m_has_result = false;
        m_is_thread_running = true;
        m_keep_running = true;

        for(int i = 0;i < RELOCALIZE_THREADS;i++)
        {
            m_relocalize_threads[i] = std::thread(&c_Relocalizer::relocalize_loop,this,i);
        }
    }

    void c_Relocalizer::thread_stop()
    {
        m_keep_running = false;
        m_ex_mutex.lock();
        m_new_current_frame_signal.notify_all();
        m_ex_mutex.unlock();

        for(int i = 0;i < RELOCALIZE_THREADS;i++)
        {
            m_relocalize_threads[i].join();
        }

        m_KF_for_relocalize.clear();
        m_current_relocalize_frame.reset();
    }

    bool c_Relocalizer::wait_relocalize_result(int millisecond)
    {
        std::unique_lock<std::mutex> lock(m_ex_mutex);
        if(m_has_result)
            return true;
        m_result_ready_signal.wait_for(lock,std::chrono::milliseconds(millisecond));
        return m_has_result;
    }

    void c_Relocalizer::get_relocalize_result(
        c_Frame* &out_keyframe,std::shared_ptr<c_Frame>& frame,int& out_frame_ID,SE3& out_frame_to_keyframe)
    {
        std::unique_lock<std::mutex> lock(m_ex_mutex);

        if(m_has_result)
        {
            out_keyframe = m_result_KF;
            out_frame_ID = m_result_frame_ID;
            out_frame_to_keyframe = m_result_frame_to_keyframe;
            frame = m_result_relocalize_frame;
        }
        else
        {
            out_keyframe = 0;
            out_frame_ID = -1;
            out_frame_to_keyframe = SE3();
            frame.reset();
        }
    }

    void c_Relocalizer::relocalize_loop(int running_index)
    {
        if(!multiThreading && running_index != 0)
            return;

        c_SE3Tracker* tracker = new c_SE3Tracker(m_width,m_height,m_K);

        std::unique_lock<std::mutex> lock(m_ex_mutex);
        while(m_keep_running)
        {
            if(m_next_index < m_max_index && m_current_relocalize_frame)
            {
                c_Frame* tmp_frame = m_KF_for_relocalize[m_next_index % m_KF_for_relocalize.size()];
                m_next_index++;
                if(tmp_frame->m_neighbors.size() <= 2)
                    continue;
                
                std::shared_ptr<c_Frame> tmp_relocalize_frame = m_current_relocalize_frame;

                lock.unlock();

                SE3 tmp_to_frame = tracker->quick_trackFrame(tmp_frame,tmp_relocalize_frame.get(),SE3());
                float good_count_percent = tracker->m_last_good_count / (tracker->m_last_good_count + tracker->m_last_bad_count);
                float good_usage = tracker->m_point_usage * good_count_percent;

                if(good_usage > relocalizationTH)
                {
                    int num_good_neighbor = 0;
                    int num_bad_neighbor = 0;

                    float best_good_usage = good_usage;

                    c_Frame* best_KF = tmp_frame;
                    SE3 best_KF_to_frame = tmp_to_frame;

                    Sim3 frame_to_world = tmp_frame->get_scaled_cam_to_world() * sim3FromSE3(tmp_to_frame.inverse(),1);

                    for(c_Frame* neighbor : tmp_frame->m_neighbors)
                    {
                        SE3 neighbor_to_frame_init = se3FromSim3(neighbor->get_scaled_cam_to_world().inverse() * frame_to_world).inverse();
                        SE3 neighbor_to_frame = tracker->quick_trackFrame(neighbor,tmp_relocalize_frame.get(),neighbor_to_frame_init);

                        float neighbor_good_count_percent = tracker->m_last_good_count / (tracker->m_last_good_count + tracker->m_last_bad_count);
                        float neighbor_good_usage = tracker->m_point_usage * neighbor_good_count_percent;

                        if(
                            neighbor_good_usage > relocalizationTH * 0.8 
                            && (neighbor_to_frame * neighbor_to_frame_init.inverse()).log().norm() < 0.1)
                        {
                            num_good_neighbor++;
                        }
                        else
                        {
                            num_bad_neighbor++;
                        }

                        if(neighbor_good_usage > best_good_usage)
                        {
                            best_good_usage = neighbor_good_usage;
                            best_KF = neighbor;
                            best_KF_to_frame = neighbor_to_frame;
                        }
                    }

                    if(num_good_neighbor > num_bad_neighbor || num_good_neighbor >= 5)
                    {
                        m_keep_running = false;
                        lock.lock();
                        m_result_relocalize_frame = tmp_relocalize_frame;
                        m_result_frame_ID = tmp_relocalize_frame->get_id();
                        m_result_KF = best_KF;
                        m_result_frame_to_keyframe = best_KF_to_frame.inverse();
                        m_result_ready_signal.notify_all();
                        m_has_result = true;
                        lock.unlock();
                    }

                }
                lock.lock();
            }
            else
                m_new_current_frame_signal.wait(lock);
        }

        delete tracker;
    }
}
