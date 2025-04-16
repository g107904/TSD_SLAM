#pragma once
#include <mutex>
#include <condition_variable>

namespace lsd_slam
{
    class c_read_write_lock
    {
    public:
        c_read_write_lock();


        void read_lock();
        void read_unlock();
        void write_lock();
        void write_unlock();
        bool try_write_lock_for(int times);
        void unlock_try_write_lock();

        void lock();
        void unlock();

        void lock_shared();
        void unlock_shared();

        bool is_should_write()
        {
            return m_read_count == 0 && !m_write_flag;
        }

        //debug
        int get_read_count()
        {
            read_lock();
            int t = m_read_count;
            read_unlock();
            return t;
        }

    private:

        std::mutex m_mutex;
        bool m_write_flag;
        int m_read_count;
        int m_write_count;
        std::condition_variable m_write_cond;
        std::condition_variable m_read_cond;
    };
}
