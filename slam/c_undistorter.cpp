#include "c_undistorter.h"
#include "util/settings.h"
#include "util/interpolation.h"

#include <sstream>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <Eigen/Core>


namespace lsd_slam
{
    c_undistorter::~c_undistorter()
    {

    }



    c_undistorter_atan::c_undistorter_atan(const char* filename)
    {
        m_is_valid = true;
        m_remap_x = nullptr;
        m_remap_y = nullptr;

        std::ifstream file(filename);

        std::string line[4];

        for(int i = 0;i < 4;i++)
            std::getline(file,line[i]);
        
        std::vector<std::string> st;

        //line 1
        read_num_from_line(st,line[0]);

        if(st.size() != 5)
        {
            printf("Fail to read!\n");
            m_is_valid = false;
            return;
        }

        for(int i = 0;i < 5;i++)
        {
            m_in_calibration[i] = (float)(atof(st[i].c_str()));
        }

        st.clear();

        //line 2
        read_num_from_line(st,line[1]);

        if(st.size() != 2)
        {
            printf("Fail to read!\n");
            m_is_valid = false;
            return;
        }

        m_in_width = atoi(st[0].c_str());
        m_in_height = atoi(st[1].c_str());

        st.clear();

        //line 3
        read_num_from_line(st,line[2]);

        if(st.size() == 1)
        {
            if(st[0][0] == 'c') //crop
                m_out_calibration[0] = -1;
            else if(st[0][0] == 'f') //full
                m_out_calibration[0] = -2;
            else if(st[0][0] == 'n') //none
                ;
            else
            {
                printf("Fail to read!\n");
                m_is_valid = false;
                return;
            }
        }
        else if(st.size() == 5)
        {
            for(int i = 0;i < 5;i++)
                m_out_calibration[i] = (float)atof(st[i].c_str());
        }
        else
        {
            printf("Fail to read!\n");
            m_is_valid = false;
            return;
        }

        st.clear();

        //line 4

         read_num_from_line(st,line[1]);

        if(st.size() != 2)
        {
            printf("Fail to read!\n");
            m_is_valid = false;
            return;
        }

        m_out_width = atoi(st[0].c_str());
        m_out_height = atoi(st[1].c_str());

        st.clear();

        float omega = m_in_calibration[4];
        float denominator = 2.0f * tan(omega / 2.0f);

        float in_fx = m_in_calibration[0] * m_in_width;
        float in_fy = m_in_calibration[1] * m_in_height;
        float in_cx = m_in_calibration[2] * m_in_width - 0.5;
        float in_cy = m_in_calibration[3] * m_in_height - 0.5;

        double x_factor = m_in_width / (1.0 * m_in_width);
        double y_factor = m_in_height / (1.0 * m_in_height);
        in_fx = in_fx * x_factor;
        in_fy = in_fy * y_factor;
        in_cx = (in_cx + 0.5) * x_factor - 0.5;
        in_cy = (in_cy + 0.5) * y_factor - 0.5;

        float out_fx,out_fy,out_cx,out_cy;

        if(omega == 0) //no FOV,pin hole camera
        {
            out_fx = m_in_calibration[0] * m_out_width;
            out_fy = m_in_calibration[1] * m_out_height;
            out_cx = m_in_calibration[2] * m_out_width - 0.5;
            out_cy = m_in_calibration[3] * m_out_height - 0.5;
        }
        else if(m_out_calibration[0] != -1 && m_out_calibration[0] != -2) // 5 out_calibration
        {
            out_fx = m_out_calibration[0] *  m_out_width;
            out_fy = m_out_calibration[1] * m_out_height;
            out_cx = m_out_calibration[2] * m_out_width - 0.5;
            out_cy = m_out_calibration[3] * m_out_height - 0.5;
        }
        else // "crop" or "full"
        {
            float left_radius = (in_cx) / in_fx;
            float right_radius = (m_in_width - 1 - in_cx) / in_fx;
            float top_radius = (in_cy) / in_fy;
            float bottom_radius = (m_in_height - 1 - in_cy)  / in_fy;

            if(m_out_calibration[0] == -1) // crop
            {
                float trans_left_radius = tan(left_radius * omega) / denominator;
                float trans_right_radius = tan(right_radius * omega) / denominator;
                float trans_top_radius = tan(top_radius * omega) / denominator;
                float trans_bottom_radius = tan(bottom_radius * omega) / denominator;

                out_fx = in_fx * ((left_radius+right_radius)/(trans_left_radius+trans_right_radius)) * ((float)m_out_width / (float)m_in_width);
                out_cx = (trans_left_radius/left_radius) * out_fx * in_cx / in_fx;

                out_fy = in_fy * ((top_radius+bottom_radius)/(trans_top_radius+trans_bottom_radius)) * ((float)m_out_height / (float)m_in_height);
                out_cy = (trans_top_radius/top_radius) * out_fy * in_cy / in_fy;
            }
            else if(m_out_calibration[0] == -2) // full
            {
                float tl_radius = sqrt(left_radius*left_radius+top_radius*top_radius);
                float tr_radius = sqrt(right_radius*right_radius+top_radius*top_radius);
                float bl_radius = sqrt(left_radius*left_radius+bottom_radius*bottom_radius);
                float br_radius = sqrt(right_radius*right_radius+bottom_radius*bottom_radius);

                float trans_tl_radius = tan(tl_radius * omega) / denominator;
                float trans_tr_radius = tan(tl_radius * omega) / denominator;
                float trans_bl_radius = tan(bl_radius * omega) / denominator;
                float trans_br_radius = tan(br_radius * omega) / denominator;

                float horizon = std::max(br_radius,tr_radius) + std::max(bl_radius,tl_radius);
                float vertical = std::max(tr_radius,tl_radius) + std::max(bl_radius,br_radius);

                float trans_horizon = std::max(trans_br_radius,trans_tr_radius) + std::max(trans_bl_radius,trans_tl_radius);
                float trans_vertical = std::max(trans_tr_radius,trans_tl_radius) + std::max(trans_bl_radius,trans_br_radius);

                out_fx = in_fx * ((horizon / trans_horizon)) * ((float)m_out_width / (float)m_in_width);
                out_cx = std::max(trans_bl_radius/bl_radius,trans_tl_radius/tl_radius) * out_fx * in_cx / in_fx;

                out_fy = in_fy * ((vertical / trans_vertical)) * ((float) m_out_height / (float)m_in_height);
                out_cy = std::max(trans_tl_radius/tl_radius,trans_tr_radius/tr_radius) * out_fy * in_cy / in_fy;
            }


        }

        m_out_calibration[0] = out_fx / m_out_width;
        m_out_calibration[1] = out_fy / m_out_height;
        m_out_calibration[2] = (out_cx + 0.5) / m_out_width;
        m_out_calibration[3] = (out_cy + 0.5) / m_out_height;
        m_out_calibration[4] = 0;

        m_remap_x = (float*) Eigen::internal::aligned_malloc(m_out_width * m_out_height * sizeof(float));
        m_remap_y = (float*) Eigen::internal::aligned_malloc(m_out_width * m_out_height * sizeof(float));

        for(int y = 0;y < m_out_height;y++)
            for(int x = 0;x < m_out_width;x++)
            {
                float ix = (x - out_cx) / out_fx;
                float iy = (y - out_cy) / out_fy;

                float r = sqrt(ix*ix+iy*iy);
                float fac = (r == 0 || omega == 0) ? 1 : atan(r * denominator) / (omega * r);

                ix = in_fx * fac * ix + in_cx;
                iy = in_fy * fac * iy + in_cy;

                if(ix == 0) ix = 0.01;
                if(iy == 0) iy = 0.01;
                if(ix == m_in_width-1) ix = m_in_width-1.01;
                if(iy == m_in_height-1) iy = m_in_height - 1.01;

                int index = x+y*m_out_width;
                if(ix > 0 && iy > 0 && ix < m_in_width-1 && iy < m_in_height-1)
                {
                    m_remap_x[index] = ix;
                    m_remap_y[index] = iy;
                }
                else
                {
                    m_remap_x[index] = -1;
                    m_remap_y[index] = -1;
                }
            } 

        m_original_K = cv::Mat(3,3,CV_64F,cv::Scalar(0));
        m_original_K.at<double>(0,0) = m_in_calibration[0];
        m_original_K.at<double>(1,1) = m_in_calibration[1];
        m_original_K.at<double>(2,2) = 1;
        m_original_K.at<double>(2,0) = m_in_calibration[2];
        m_original_K.at<double>(2,1) = m_in_calibration[3];

        m_K = cv::Mat(3,3,CV_64F,cv::Scalar(0));
        m_K.at<double>(0,0) = m_out_calibration[0] * m_out_width;
        m_K.at<double>(1,1) = m_out_calibration[1] * m_out_height;
        m_K.at<double>(2,2) = 1;
        m_K.at<double>(2,0) = m_out_calibration[2] * m_out_width - 0.5;
        m_K.at<double>(2,1) = m_out_calibration[3] * m_out_height - 0.5;
    }

    c_undistorter_atan::~c_undistorter_atan()
    {
        Eigen::internal::aligned_free((void*)m_remap_x);
        Eigen::internal::aligned_free((void*)m_remap_y);
    }

    void c_undistorter_atan::undistort(const cv::Mat& image,cv::OutputArray result) const
    {
        if(!m_is_valid)
        {
            result.getMatRef() = image;
            return;
        }

        if(image.rows != m_in_height || image.cols != m_in_width)
        {
            result.getMatRef() = image;
            return;
        }

        if(m_in_height == m_out_height && m_in_width == m_out_width && m_in_calibration[4] == 0)
        {
            result.getMatRef() = image;
            return;
        }

        result.create(m_out_height,m_out_width,CV_8U);
        cv::Mat result_mat = result.getMatRef();

        assert(result.getMatRef().isContinuous());
	    assert(image.isContinuous());

        unsigned char* data = result_mat.data;

        for(int index = m_out_width * m_out_height - 1;index > 0;index--)
        {
            float ix = m_remap_x[index];
            float iy = m_remap_y[index];

            if(ix < 0)
                data[index] = 0;
            else
            {
                data[index] = get_interpolated_element((unsigned char*)image.data,ix,iy,m_in_width);
            }
        }
    }

    c_undistorter_opencv::c_undistorter_opencv(const char* filename)
    {
        m_is_valid = true;

        std::ifstream file(filename);
        assert(file.good());

        std::string line[4];
        
        for(int i = 0;i < 4;i++)
            std::getline(file,line[i]);
        
        std::vector<std::string> st;

        //line 1
        read_num_from_line(st,line[0]);

        if(st.size() < 8)
        {
            printf("Fail to read!\n");
            m_is_valid = false;
            return;
        }

        for(int i = 0;i < 8;i++)
        {
            m_in_calibration[i] = (float)(atof(st[i].c_str()));
        }

        st.clear();

        //line 2
        read_num_from_line(st,line[1]);

        if(st.size() != 2)
        {
            printf("Fail to read!\n");
            m_is_valid = false;
            return;
        }

        m_in_width = atoi(st[0].c_str());
        m_in_height = atoi(st[1].c_str());

        st.clear();

        //line 3
        read_num_from_line(st,line[2]);

        if(st.size() == 1)
        {
            if(st[0][0] == 'c') //crop
                m_out_calibration = -1;
            else if(st[0][0] == 'f') //full
                m_out_calibration = -2;
            else if(st[0][0] == 'n') //none
            {
                printf("Fail to read!\n");
                m_is_valid = false;
                return;
            }
            else
            {
                printf("Fail to read!\n");
                m_is_valid = false;
                return;
            }
        }
        else
        {
            printf("Fail to read!\n");
            m_is_valid = false;
            return;
        }

        st.clear();

        //line 4

         read_num_from_line(st,line[1]);

        if(st.size() != 2)
        {
            printf("Fail to read!\n");
            m_is_valid = false;
            return;
        }

        m_out_width = atoi(st[0].c_str());
        m_out_height = atoi(st[1].c_str());

        st.clear();

        cv::Mat dist_coeffs = cv::Mat::zeros(4,1,CV_64F);
        for(int i = 0;i < 4;i++)
        {
            dist_coeffs.at<double>(i,0) = m_in_calibration[4+i];
        }

        if(m_in_calibration[2] < 1.0f)
        {//unusual
            m_in_calibration[0] *= m_in_width;
            m_in_calibration[2] *= m_in_width;
            m_in_calibration[1] *= m_in_height;
            m_in_calibration[3] *= m_in_height;
        }

        m_original_K = cv::Mat(3,3,CV_64F,cv::Scalar(0));
        m_original_K.at<double>(0,0) = m_in_calibration[0];
        m_original_K.at<double>(1,1) = m_in_calibration[1];
        m_original_K.at<double>(2,2) = 1;
        m_original_K.at<double>(0,2) = m_in_calibration[2];
        m_original_K.at<double>(1,2) = m_in_calibration[3];

        

        m_K = cv::getOptimalNewCameraMatrix(m_original_K,dist_coeffs,cv::Size(m_in_width,m_in_height),(m_out_calibration == -2)? 1 : 0,cv::Size(m_out_width,m_out_height),nullptr,false);

        cv::initUndistortRectifyMap(m_original_K,dist_coeffs,cv::Mat(),m_K,cv::Size(m_out_width,m_out_height),CV_16SC2,m_map1,m_map2);

        m_original_K.at<double>(0,0) /= m_in_width;
        m_original_K.at<double>(0,2) /= m_in_width;
        m_original_K.at<double>(1,1) /= m_in_height;
        m_original_K.at<double>(1,2) /= m_in_height;
        
        m_original_K = m_original_K.t();
        m_K = m_K.t();
    }

    c_undistorter_opencv::~c_undistorter_opencv()
    {

    }

    void c_undistorter_opencv::undistort(const cv::Mat& image,cv::OutputArray result) const
    {
        cv::remap(image,result,m_map1,m_map2,cv::INTER_LINEAR);
    }


    c_undistorter* c_undistorter::get_undistorter_from_file(const char* filename)
    {
        std::ifstream file(filename);
        if (!file.good())
        {
            file.close();
            printf("not found calib file!\n");
            return 0;
        }

        std::string s;
        std::getline(file, s);
        file.close();

        std::vector<std::string> st;

        read_num_from_line(st, s);

        if (st.size() >= 8)
        {
            c_undistorter* undistorter = new c_undistorter_opencv(filename);
            if (!undistorter->get_valid())
                return 0;
            return undistorter;
        }
        else
        {
            c_undistorter* undistorter = new c_undistorter_atan(filename);
            if (!undistorter->get_valid())
                return 0;
            return undistorter;
        }
    }
}
