#include "util/settings.h"
#include "util/interpolation.h"
#include "c_SlamSystem.h"

#include "opencv2/opencv.hpp"

#include <stdio.h>
#include <vector>
#include <thread>
#include <algorithm>


/*
#ifdef _WIN32
#include <io.h>
#define get_filenames get_filenames_win32
void get_filenames_win32(std::string image_path, std::vector<std::string>& image_files)
{
    long file = 0;
    struct _finddata_t file_info;

    std::string tmp;

    if ((file = _findfirst(tmp.assign(image_path).append("\\*").c_str(), &file_info)) != -1)
    {
        do {
            if (file_info.name[0] == '.')
                continue;
            image_files.push_back(file_info.name);
        } while (_findnext(file, &file_info) == 0);

        _findclose(file);

        std::sort(image_files.begin(), image_files.end());
        image_files.pop_back();

        if (image_path.at(image_path.length() == 1) != '/')
            image_path = image_path + "\\";

        for (int i = 0; i < image_files.size(); i++)
        {
            image_files[i] = image_path + image_files[i];
        }

    }
}


#endif

#ifdef __GNUC__
#include <dirent.h>
#define get_filenames get_filenames_posix
void get_filenames_posix(std::string image_path, std::vector<std::string>& image_files)
{
    DIR* tmp_dir;
    struct dirent* dir_pt;
    if ((tmp_dir = opendir(image_path.c_str())) == nullptr)
    {
        return;
    }

    while ((dir_pt = readdir(tmp_dir)) != nullptr)
    {
        std::string filename = std::string(dir_pt->d_name);

        if (filename != "." && filename != "..")
        {
            image_files.push_back(filename);
        }
    }
    closedir(tmp_dir);

    std::sort(image_files.begin(), image_files.end());

    if (image_path.at(image_path.length() - 1) != '/')
        image_path = image_path + "/";

    for (int i = 0; i < image_files.size(); i++)
    {
        if (image_files[i].at(0) != '/')
        {
            image_files[i] = image_path + image_files[i];
        }
    }
}
#endif

using namespace lsd_slam;



int main(int argc, char** argv)
{
    std::string calib_file;
    std::string image_path;
    std::string equation = "=";
    std::vector<std::string> image_files;
    for (int i = 0; i < argc; i++)
    {
        if (argv[i][0] == 'c')
        {
            std::string tmp_s = argv[i];
            int pos = tmp_s.find(equation);
            calib_file = tmp_s.substr(pos + 1, tmp_s.length() - pos - 1);
        }
        if (argv[i][0] == 'f')
        {
            std::string tmp_s = argv[i];
            int pos = tmp_s.find(equation);
            image_path = tmp_s.substr(pos + 1, tmp_s.length() - pos - 1);
        }
    }
    get_filenames(image_path, image_files);


    //c_undistorter* undistorter = c_undistorter::get_undistorter_from_file(calib_file.c_str());



    int out_w = 1080;
    int out_h = 720;

    int in_w = 1080;
    int in_h = 720;

    Eigen::Matrix3f K;
    K.setZero();
    float f = 8.8 * 0.001;
    float cmos_w = 12.8 * 0.001;
    float cmos_h = 9.6 * 0.001;
    float fx = f / cmos_w * out_w;
    float cx = out_w / 2;
    float fy = f / cmos_h * out_h;
    float cy = out_h / 2;

    K(0, 0) = fx;
    K(0, 2) = cx;
    K(1, 1) = fy;
    K(1, 2) = cy;
    K(2, 2) = 1;
    K << fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0;


    c_SlamSystem* system = new c_SlamSystem(out_w, out_h, K);




    cv::Mat image = cv::Mat(out_h, out_w, CV_8U);

    for (int i = 0; i < image_files.size(); i++)
    {
        std::cout << i << std::endl;
        cv::Mat image_raw = cv::imread(image_files[i], CV_LOAD_IMAGE_GRAYSCALE);

        if (image_raw.rows != in_h || image_raw.cols != in_w)
        {
            continue;
        }

        //undistorter->undistort(image_raw, image);
        image = image_raw;

        if (i == 0)
            system->random_init(image.data, i);
        else
            system->trackFrame(image.data, i);


    }
    system->finalize();


    delete system;
    //delete undistorter;

    return 0;


}
*/

class test {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    void* use_test(size_t size)
    {
        void* buffer = Eigen::internal::aligned_malloc(size);
        return buffer;
    }
};

int main()
{
    test t;
    void* q = t.use_test(12060);
    void* w = t.use_test(48240);
}