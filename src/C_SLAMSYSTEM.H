#pragma once


#include <mutex>
#include <condition_variable>
#include <thread>

#include <Eigen/Core>
#include<opencv2/core.hpp>
#include "gl/c_slam_draw.h"

namespace lsd_slam
{
    

    class c_DepthMapEstimation;
    class c_MapOptimization;
    class c_SE3Tracker;
    

    class c_SlamSystem
    {
        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

            int m_width,m_height;
            Eigen::Matrix3f m_K;
            std::vector<std::string> m_image_files;

            __declspec(dllexport) c_SlamSystem(int w,int h,Eigen::Matrix3f K);
            

            __declspec(dllexport) void  random_init(unsigned char* image,int id);
            __declspec(dllexport) void  get_depth_init(unsigned char* image, float* depth,int id);

            __declspec(dllexport) void  trackFrame(cv::Mat image,unsigned int frame_id);

            SE3 feature_use(cv::Mat origin,cv::Mat cur,const float* origin_idepth);

            void constraint_thread_loop();

            void optimization_thread_loop();

            void slam_gl_change_thread_loop();

            void map_thread_loop();
            
            __declspec(dllexport) void finalize();
            __declspec(dllexport) ~c_SlamSystem();

        private:
            c_DepthMapEstimation* m_depthmap_estimation;
            c_MapOptimization* m_map_optimization;
            c_SE3Tracker* m_tracker;


            bool m_is_thread_keep_running;
            std::thread m_optimization_thread;
            std::thread m_constraint_thread;

            std::thread m_map_thread;

            std::thread m_slam_gl_change_thread;
            std::thread m_slam_gl_draw_thread;
            slam_gl::c_slam_draw* m_slam_draw_pt;

            
            


    };
}
