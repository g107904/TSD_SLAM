#include <Eigen/Core>
#include <vector>
#include <opencv/cv.hpp>
class c_PointLineBA
{
public:
	std::vector<cv::Mat> m_file_img;
	Eigen::Matrix3f m_K;

	c_PointLineBA() {}
	void receive_frame(cv::Mat img)
	{
		m_file_img.push_back(img);
	}

	void set_K(Eigen::Matrix3f& K)
	{
		m_K = K;
	}

	int do_BA();
};