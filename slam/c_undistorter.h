#ifndef _C_UNDISTORTER_HPP_
#define _C_UNDISTORTER_HPP_

#include <opencv2/core/core.hpp>

#include <vector>
#include <string>

#include <fstream>

namespace lsd_slam
{



    class c_undistorter
    {
        public:
            virtual ~c_undistorter();



            

            virtual void undistort(const cv::Mat& image,cv::OutputArray result) const = 0;
            
            __declspec(dllexport) static c_undistorter*  get_undistorter_from_file(const char* filename);

             static void read_num_from_line(std::vector<std::string>& st,std::string s)
                    {
                        int pos = 0;
                        while ((pos = s.find(' ')) != s.npos)
                        {

                            st.push_back(s.substr(0, pos ));
                            s = s.substr(pos+1,s.length()-pos-1);
                        }
                        st.push_back(s);
                    }

            const cv::Mat& get_K() const 
            {
                return m_K;
            }

            const cv::Mat& get_original_K() const
            {
                return m_original_K;
            } 

            int get_out_width() const
            {
                return m_out_width;
            }

            int get_out_height() const
            {
                return m_out_height;
            }

            int get_in_width() const 
            {
                return m_in_width;
            }
            
            int get_in_height() const
            {
                return m_in_height;
            }

            int get_valid() const
            {
                return m_is_valid;
            }
            
        protected:
            cv::Mat m_K;
            cv::Mat m_original_K;

            int m_out_width,m_out_height;
            int m_in_width,m_in_height;

            bool m_is_valid;
    };

    class c_undistorter_atan : public c_undistorter
    {
        public:
            c_undistorter_atan(const char* filename);

            ~c_undistorter_atan();

            void undistort(const cv::Mat& image,cv::OutputArray result)const;
            
            c_undistorter_atan(const c_undistorter_atan&) = delete;
            c_undistorter_atan& operator = (const c_undistorter_atan&) = delete;

        private:

            float m_in_calibration[5];
            float m_out_calibration[5];
            float* m_remap_x;
            float* m_remap_y;
    };

    class c_undistorter_opencv : public c_undistorter 
    {
        public:
            c_undistorter_opencv(const char* filename);

            ~c_undistorter_opencv();

            void undistort(const cv::Mat& image,cv::OutputArray result)const;

            c_undistorter_opencv(const c_undistorter_opencv&) = delete;

            c_undistorter_opencv& operator=(const c_undistorter_opencv&) = delete;

        private:
            float m_in_calibration[10];
            float m_out_calibration;
            cv::Mat m_map1,m_map2;
    };


}

#endif
