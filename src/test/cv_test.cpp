#pragma once 
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include "Tracking/c_SE3Tracker.h"
namespace lsd_slam
{
    cv::Mat show_it_image(int id, Sophus::SE3 ref_to_frame, slam::c_Frame* frame1, slam::c_frame* frame2, int level = 0)
    {

        int width = frame1->get_width(level);
        int height = frame1->get_height(level);
        Eigen::Matrix3f K = frame1->get_K(level);
        float* tmp = (float*)malloc(sizeof(float) * width * height);
        memset(tmp, 0, sizeof(float) * width * height);
        frame1->m_pointcloud->make_pointcloud(level);
        const float* idepth = frame1->get_idepth(level);
        const float* image = frame1->get_image(level);
        Eigen::Vector3f* point_3d = frame1->m_pointcloud->m_3d_pos_data[level];
        int* point_2d = frame1->m_pointcloud->m_2d_index[level];
        int num = frame1->m_pointcloud->m_num[level];
        for (int i = 0; i < num; i++)
        {
            int index = point_2d[i];
            Eigen::Matrix3f rotation = ref_to_frame.rotationMatrix();
            Eigen::Vector3f translation = ref_to_frame.translation();
            Eigen::Vector3f p_other = K * (rotation * point_3d[i] / idepth[index] + translation);
            float z_other = p_other[2];
            float u_new = p_other[0] / z_other;
            float v_new = p_other[1] / z_other;
            int u_new_int = (int)u_new;
            int v_new_int = (int)v_new;
            if (u_new - u_new_int > 0.5) u_new_int++;
            if (v_new - v_new_int > 0.5) v_new_int++;
            if (u_new_int < 0 || v_new_int < 0 || u_new_int >= width || v_new_int >= height)
                continue;
            tmp[u_new_int + v_new_int * width] = image[index];
        }
        int show_width = 200;
        int show_init = 200, show_end = 400;
        int color_thres = 150;

        const float* frame2_image = frame2->get_image(level);
        cv::Mat tmp_image = cv::Mat::zeros(height, width, CV_8UC3);
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
            {
                int index = x + y * width;
                int new_index = (x - show_init) + (y - show_init) * show_width;
                new_index = index;

                //tmp_image.data[new_index * 3] = tmp[index];


                if (tmp[index] > color_thres)
                {
                    tmp_image.data[new_index * 3] = 255;

                }

                if (frame2_image[index] > color_thres)
                {
                    tmp_image.data[new_index * 3 + 1] = 255;
                }


            }
        cv::namedWindow("test", 0);
        cv::imshow("test", tmp_image);
        std::string path = "D:\\slam_all\\init\\result\\rotation\\opt_origin_";

        cv::imwrite((path + std::to_string(id) + ".png").c_str(), tmp_image);
        cv::waitKey();
        return tmp_image;
    }
}