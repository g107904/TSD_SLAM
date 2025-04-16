#pragma once
#include "util/settings.h"
#include <stdio.h>
#include <iostream>
#include "util/Sophus_util.h"
#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>
#include<memory>

namespace lsd_slam
{
    class c_Frame;
    class c_Sim3Tracker;
    class c_Relocalizer
    {
        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
            c_Relocalizer(int w,int h,Eigen::Matrix3f K);
            ~c_Relocalizer();
            void update_current_frame(std::shared_ptr<c_Frame> current_frame);
            void thread_start(std::vector<c_Frame* > & keyframe_all);
            void thread_stop();

            bool wait_relocalize_result(int milliseconds);
            void get_relocalize_result(
                c_Frame* &out_keyframe,std::shared_ptr<c_Frame>& frame,int &out_frame_ID,SE3& out_frame_to_keyframe);

            bool m_is_thread_running;

        private:
            int m_width,m_height;
            Eigen::Matrix3f m_K;
            std::thread m_relocalize_threads[RELOCALIZE_THREADS];

            std::mutex m_ex_mutex;
            std::condition_variable m_new_current_frame_signal;
            std::condition_variable m_result_ready_signal;

            std::vector<c_Frame*> m_KF_for_relocalize;
            std::shared_ptr<c_Frame> m_current_relocalize_frame;
            int m_next_index;
            int m_max_index;
            bool m_keep_running;

            std::shared_ptr<c_Frame> m_result_relocalize_frame;
            bool m_has_result;
            c_Frame* m_result_KF;
            int m_result_frame_ID;
            SE3 m_result_frame_to_keyframe;

            void relocalize_loop(int running_index);

    };
}