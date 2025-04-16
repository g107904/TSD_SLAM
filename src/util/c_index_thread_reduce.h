#pragma once
#include "settings.h"

#include <iostream>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <vector>
namespace lsd_slam
{
	class c_index_thread_reduce
	{
	public:
		inline c_index_thread_reduce(int num);
		inline ~c_index_thread_reduce();
		inline void reduce(std::function<void(int, int)> func, int first, int end, int step);
	private:
		std::function<void(int,int)> m_func_pt;
		int m_thread_num;
		bool m_is_runing;
		std::vector<std::thread> m_thread_pool;
		std::vector<bool> m_thread_done_flag;

		std::mutex m_mutex;
		std::condition_variable m_next_signal;
		std::condition_variable m_end_signal;

		int m_max_id;
		int m_step;
		int m_next_id;

		void thread_loop(int id);
		void default_func(int, int);
	};

	c_index_thread_reduce::c_index_thread_reduce(int num)
	{
		m_next_id = 0;
		m_max_id = 0;
		m_step = 1;

		m_is_runing = true;
		m_thread_num = num;

		m_func_pt = std::bind(&c_index_thread_reduce::default_func,this, std::placeholders::_1, std::placeholders::_2);
		
		for (int i = 0; i < num; i++)
		{
			m_thread_pool.push_back(std::thread(&c_index_thread_reduce::thread_loop, this, i));
			m_thread_done_flag.push_back(false);
		}
	}

	c_index_thread_reduce::~c_index_thread_reduce()
	{
		m_is_runing = false;
		m_mutex.lock();
		m_next_signal.notify_all();
		m_mutex.unlock();
		for (int i = 0; i < m_thread_num; i++)
			m_thread_pool[i].join();
		m_thread_pool.swap(std::vector<std::thread>());
		m_thread_done_flag.swap(std::vector<bool>());
	}

	void c_index_thread_reduce::reduce(std::function<void(int, int)> func, int first, int end, int step)
	{
		if (!multiThreading)
		{
			func(first, end);
			return;
		}

		if (step == 0)
			step = (end - first + m_thread_num - 1) / m_thread_num;

		std::unique_lock<std::mutex> lock(m_mutex);
		this->m_func_pt = func;
		m_next_id = first;
		m_max_id = end;
		m_step = step;

		for (int i = 0; i < m_thread_num; i++)
			m_thread_done_flag[i] = false;
		m_next_signal.notify_all();

		while (true)
		{
			m_end_signal.wait(lock);
			bool flag = true;
			for (int i = 0; i < m_thread_num; i++)
				flag = flag && m_thread_done_flag[i];
			if (flag)
				break;
		}

		m_max_id = m_next_id = 0;
		this->m_func_pt = std::bind(&c_index_thread_reduce::default_func, this, std::placeholders::_1, std::placeholders::_2);
	}

	void c_index_thread_reduce::thread_loop(int id)
	{
		std::unique_lock<std::mutex> lock(m_mutex);
		while (m_is_runing)
		{
			int do_id = 0;
			bool should_do = false;
			if (m_next_id < m_max_id)
			{
				do_id = m_next_id;
				m_next_id += m_step;
				should_do = true;
			}
			if (should_do)
			{
				lock.unlock();
				m_func_pt(do_id, std::min(do_id + m_step, m_max_id));
				lock.lock();
			}
			else
			{
				m_thread_done_flag[id] = true;
				m_end_signal.notify_all();
				m_next_signal.wait(lock);
			}
		}

	}

	void c_index_thread_reduce::default_func(int i, int j)
	{
		std::cout << "error!" << std::endl;
	}
}