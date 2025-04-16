#include "util/c_read_write_lock.h"
#include <thread>


namespace lsd_slam
{
    c_read_write_lock::c_read_write_lock()
    {
        m_write_flag = false;
        m_read_count = 0;
        m_write_count = 0;
    }



    void c_read_write_lock::read_lock()
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_read_cond.wait(lock, [this]()->bool {return m_write_count == 0; });
        ++m_read_count;
    }

    void c_read_write_lock::read_unlock()
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        if (--m_read_count == 0 && m_write_count > 0)
        {
            m_write_cond.notify_one();
        }

    }

    void c_read_write_lock::write_lock()
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        ++m_write_count;
        m_write_cond.wait(lock, [this]()-> bool {return m_read_count == 0 && !m_write_flag; });
        m_write_flag = true;
    }



    void c_read_write_lock::write_unlock()
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        if (--m_write_count == 0)
        {
            m_read_cond.notify_all();
        }
        else
        {
            m_write_cond.notify_one();
        }
        m_write_flag = false;
    }

    bool c_read_write_lock::try_write_lock_for(int times)
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        ++m_write_count;
        bool flag = m_write_cond.wait_for(lock, std::chrono::milliseconds(times), [this]()-> bool {return m_read_count == 0 && !m_write_flag; });
        if (flag)
        {
            m_write_flag = true;
        }
        else
        {
            --m_write_count;
        }
        return flag;
    }

    void c_read_write_lock::unlock_try_write_lock()
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        if (--m_write_count == 0)
        {
            m_read_cond.notify_all();
        }
        else
        {
            m_write_cond.notify_one();
        }
        m_write_flag = false;
    }

    void c_read_write_lock::lock()
    {
        write_lock();
    }

    void c_read_write_lock::unlock()
    {
        write_unlock();
    }

    void c_read_write_lock::lock_shared()
    {
        read_lock();
    }
    void c_read_write_lock::unlock_shared()
    {
        read_unlock();
    }
}
