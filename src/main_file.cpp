#include "util/settings.h"
#include "util/interpolation.h"
#include "c_SlamSystem.h"

#include "opencv2/opencv.hpp"
#include "../my_code_2019/slam_test/c_undistorter.h"

#include <stdio.h>
#include <vector>
#include <thread>
#include <algorithm>



#ifdef _WIN32
#include <io.h>
#define get_filenames get_filenames_win32
void get_filenames_win32(std::string image_path,std::vector<std::string>& image_files)
{
    long file = 0;
    struct _finddata_t file_info;
    
    std::string tmp;

    if((file = _findfirst(tmp.assign(image_path).append("\\*").c_str(),&file_info)) != -1)
    {
        do{
            if (file_info.name[0] == '.')
                continue;
            image_files.push_back(file_info.name);
        } while(_findnext(file,&file_info) == 0);

        _findclose(file);

        std::sort(image_files.begin(),image_files.end());
        image_files.pop_back();
    
        if(image_path.at(image_path.length()==1) != '/' )
            image_path = image_path + "\\";

        for(int i = 0;i  < image_files.size();i++)
        {
            image_files[i] = image_path + image_files[i];
        }

    }
}


#endif

#ifdef __GNUC__
#include <dirent.h>
#define get_filenames get_filenames_posix
void get_filenames_posix(std::string image_path,std::vector<std::string>& image_files)
{
    DIR* tmp_dir;
    struct dirent* dir_pt;
    if((tmp_dir = opendir(image_path.c_str())) == nullptr)
    {
        return;
    } 

    while((dir_pt = readdir(tmp_dir)) != nullptr)
    {
        std::string filename =  std::string(dir_pt->d_name);

        if(filename != "." && filename != "..")
        {
            image_files.push_back(filename);
        }
    }
    closedir(tmp_dir);

    std::sort(image_files.begin(),image_files.end());
    
    if(image_path.at(image_path.length()-1) != '/' )
        image_path = image_path + "/";

    for(int i = 0;i  < image_files.size();i++)
    {
        if(image_files[i].at(0) != '/')
        {
            image_files[i] = image_path + image_files[i];
        }
    }
}
#endif

using namespace lsd_slam;

void get_dic(std::vector<std::string>& image_files)
{
    std::string dic_files = "D:\\slam_all\\init\\dic.txt";
    FILE* fp = fopen(dic_files.c_str(), "w+");
    for (int i = 0; i < image_files.size(); i++)
    {
        std::string str = image_files[i];
        int length = str.length();
        int pos = -1;
        for (int j = length - 1; j >= 0; j--)
            if (str[j] == '\\')
            {
                pos = j;
                break;
            }
        std::string id = str.substr(pos + 1, length - pos - 5);
        fprintf(fp, "%d %s\n", i, id.c_str());
    }
    fclose(fp);
}


int main(int argc,char** argv)
{
    std::string calib_file;
    std::string image_path;
    std::string equation = "=";
    std::vector<std::string> image_files;
    for(int i = 0;i < argc;i++)
    {
        if(argv[i][0]=='c')
		{
			std::string tmp_s = argv[i];
			int pos = tmp_s.find(equation);
			calib_file = tmp_s.substr(pos+1,tmp_s.length()-pos-1);
		}
		if(argv[i][0] == 'f')
		{
			std::string tmp_s = argv[i];
			int pos = tmp_s.find(equation);
			image_path = tmp_s.substr(pos+1,tmp_s.length()-pos-1);
		}
    }
    get_filenames(image_path,image_files);

    
    c_undistorter* undistorter = c_undistorter::get_undistorter_from_file(calib_file.c_str());

    

    int out_w = undistorter->get_out_width();
    int out_h = undistorter->get_out_height();
    
    int in_w = undistorter->get_in_width();
    int in_h = undistorter->get_in_height();

    float fx = undistorter->get_K().at<double>(0,0);
    float fy = undistorter->get_K().at<double>(1, 1);
	float cx = undistorter->get_K().at<double>(2, 0);
	float cy = undistorter->get_K().at<double>(2, 1);
	Eigen::Matrix3f K;
	K << fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0;
    

    /*
    Eigen::Matrix3f K;
    K.setZero();
    K(2, 2) = 1;

    int out_w =480, in_w = 480;
    int out_h =640, in_h = 640;
    */

    c_SlamSystem* system = new c_SlamSystem(out_w,out_h,K);

    system->m_image_files = image_files;

    get_dic(image_files);

    cv::Mat image = cv::Mat(out_h,out_w,CV_8U);

    /*
    std::string depth_file = "D:\\download\\TUM\\rgbd_dataset_freiburg2_pioneer_slam2\\rgbd_dataset_freiburg2_pioneer_slam2\\depth\\1311877812.987032.png";
    cv::Mat depth_img = cv::imread(depth_file);
    float* depth_data = (float*)malloc(sizeof(float) * out_h * out_w);
    memset(depth_data, 0, sizeof(float) * out_h * out_w);
    for(int v = 0;v < out_h;v++)
        for (int u = 0; u < out_w; u++)
        {
            float z = depth_img.at<uchar>(v, u) /255.0f;
            int index = u + v * out_w;
            depth_data[index] = z;
        }
    */
    std::vector<double> times;
    for(int i = 0;i < image_files.size();i++)
    {     
        /*
        if (i % 2 != 0)
            continue;
        */
        
        //if (i < 2300)continue;
        if (i > 2320 && i < 2340)
            continue;
        std::cout<<i<<' '<<image_files[i]<<std::endl;

        cv::Mat image_raw = cv::imread(image_files[i]);

        cv::Mat img;
        if (image_raw.channels() == 3)
            cv::cvtColor(image_raw, img, cv::COLOR_BGR2GRAY);
        else
            img = image_raw;
        
        if (image_raw.rows > 1000)
        {
            image_raw = img;
            cv::resize(image_raw, img, cv::Size(image_raw.cols / 2,image_raw.rows/2));
            
        }
        
        image_raw = img;
        /*
        if(image_raw.rows != in_h || image_raw.cols != in_w)
        {
            continue;
        }
        */

        undistorter->undistort(image_raw,image);
        //cv::imshow("img", image);
        //cv::waitKey();
        //image = image_raw;

        if (i == 0)
        {
            system->random_init(image.data, i);
            //system->get_depth_init(image.data, depth_data, i);
        }
        else
        {
            auto t_start = std::chrono::steady_clock::now();
            system->trackFrame(image, i);
            auto t_end = std::chrono::steady_clock::now();
            double dr_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
            std::cout << dr_ms << std::endl;
            times.push_back(dr_ms);
        }

        /*
        if (i > 30)
            break;
            */
        

    }

    /*
    std::string out_time_file = "D:\\slam_all\\init\\large_time.txt";
    FILE* fp = fopen(out_time_file.c_str(),"w+");
    for (int i = 0; i < times.size(); i++)
    {
        fprintf(fp, "%f\n", times[i]);
    }
    fclose(fp);
    */

    system->finalize();


    delete system;
    //delete undistorter;

    return 0;

    
}
