#pragma once
#include <unordered_map>
#include <vector>
#include <mutex>
#include <deque>
#include <list>
#include <Eigen/Core>

#include "util/c_read_write_lock.h"

namespace lsd_slam
{
    class c_Frame;
    class c_FrameMemory
    {
        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

            static c_FrameMemory& get_instance();

            void* dispatch_buffer(unsigned int byte_size);

            void reclaim_buffer(void* buffer_ptr);

            c_read_write_lock* activate_frame(c_Frame* frame);

            void deactivate_frame(c_Frame* frame);

            void prune_active_frames();

            void release_buffers();

        private:
            c_FrameMemory();

            //static c_FrameMemory instance;

            std::mutex m_access_mutex;

            std::unordered_map<void*,unsigned int> m_size_of_buffer;
            std::unordered_map<unsigned int,std::vector<void*> > m_available_buffers;

            std::mutex m_active_frames_mutex;
            std::list<c_Frame*> m_active_frames;
    };
}
