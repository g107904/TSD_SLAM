#include "DataStructures/c_FrameMemory.h"
#include "DataStructures/c_Frame.h"


namespace lsd_slam
{
    c_FrameMemory::c_FrameMemory()
    {
    }

    c_FrameMemory& c_FrameMemory::get_instance()
    {
        static c_FrameMemory instance;
        return instance;
    }

    void* c_FrameMemory::dispatch_buffer(unsigned int byte_size)
    {
        if (byte_size % 1024 != 0)
            byte_size = (byte_size / 1024 + 1) * 1024;

        std::unique_lock<std::mutex> lock(m_access_mutex);

        if(m_available_buffers.count(byte_size) > 0)
        {
            std::vector<void*>& buffer_vector = m_available_buffers.at(byte_size);

            if(buffer_vector.empty())
            {
                void* buffer = Eigen::internal::aligned_malloc(byte_size);
                memset(buffer,0,byte_size);
                m_size_of_buffer.insert(std::make_pair(buffer,byte_size));
                return buffer;
            }
            else
            {
                void* buffer = buffer_vector.back();
                buffer_vector.pop_back();
                return buffer;
            }
        }
        else
        {
            void* buffer = Eigen::internal::aligned_malloc(byte_size);
            memset(buffer,0,byte_size);
            m_size_of_buffer.insert(std::make_pair(buffer,byte_size));
            return buffer;
        }
    }

    void c_FrameMemory::reclaim_buffer(void* buffer_ptr)
    {
        if(buffer_ptr == 0)
            return;
        
        std::unique_lock<std::mutex> lock(m_access_mutex);

        unsigned int buffer_size = m_size_of_buffer.at(buffer_ptr);

        memset(buffer_ptr,0,buffer_size);
        
        if(m_available_buffers.count(buffer_size) > 0)
        {
            m_available_buffers.at(buffer_size).push_back(buffer_ptr);
        }
        else
        {
            std::vector<void*> tmp_buffer_vector;
            tmp_buffer_vector.push_back(buffer_ptr);

            m_available_buffers.insert(std::make_pair(buffer_size,tmp_buffer_vector));
        }
    }

    c_read_write_lock* c_FrameMemory::activate_frame(c_Frame* frame)
    {
        std::unique_lock<std::mutex> lock(m_active_frames_mutex);

        if(frame->m_is_active)
            m_active_frames.remove(frame);
        m_active_frames.push_front(frame);
        frame->m_is_active = true;

        frame->m_active_mutex.lock_shared();
	return &(frame->m_active_mutex);
    }

    void c_FrameMemory::deactivate_frame(c_Frame* frame)
    {
        std::unique_lock<std::mutex> lock(m_active_frames_mutex);

        if(!frame->m_is_active)
            return;

        m_active_frames.remove(frame);

        while(!frame->minimize_memory());

        frame->m_is_active = false;
    }

    void c_FrameMemory::prune_active_frames()
    {
        std::unique_lock<std::mutex> lock(m_active_frames_mutex);

        while(m_active_frames.size() > maxLoopClosureCandidates+20)
        {
            if(!m_active_frames.back()->minimize_memory())
            {
                if(!m_active_frames.back()->minimize_memory())
                {
                    return;
                }
            }

            m_active_frames.back()->m_is_active = false;
            m_active_frames.pop_back();
        }
    }

    void c_FrameMemory::release_buffers()
    {
        std::unique_lock<std::mutex> lock(m_access_mutex);

        for(auto it : m_available_buffers)
        {
            for(int i = 0;i < it.second.size();i++)
            {
                Eigen::internal::aligned_free(it.second[i]);
                m_size_of_buffer.erase(it.second[i]);
            }

            it.second.clear();
        }

        m_available_buffers.clear();
    }

}
