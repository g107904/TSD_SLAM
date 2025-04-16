#pragma once
#include "FabMap.h"
#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <map>
#include <opencv2/line_descriptor.hpp>
#include <opencv2/features2d.hpp>
#include "c_PL_AdaLAM.hpp"
#include <Eigen/LU>
#include <Eigen/Dense>
#include "sophus/sophus.hpp"
#include "sophus/se3.hpp"
#include "world_element.h"
#include "g2o_ba.h"
#include "c_PointLineBA.h"

void test_lsd_opencv_core()
{
	cv::Mat src_img = cv::imread("D:\\download\\EUROC\\MH_01_easy\\mav0\\cam0\\data\\1403636579763555584.png", CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat img = src_img;
	auto lsd = cv::createLineSegmentDetector(cv::LSD_REFINE_STD);
	std::vector<cv::Vec4f> vecLines;
	lsd->detect(img, vecLines);
	/*
	lsd->drawSegments(img, vecLines);
	cv::imshow("reslineMat", img);
	cv::waitKey();
	*/
	std::vector<float> lines_k;
	for (int i = 0; i < vecLines.size(); i++)
	{
		float tmp = (vecLines[i][3] - vecLines[i][1]) / (vecLines[i][2] - vecLines[i][0] + 1e-8);
		lines_k.push_back(tmp);
	}
	std::map<float, int> ma;
	for (int i = 0; i < lines_k.size(); i++)
	{
		float key = abs(lines_k[i]) - 1e-3;
		auto pos = ma.lower_bound(key);
		if (pos == ma.end())
		{
			ma[abs(lines_k[i])] = 1;
		}
		else
			pos->second++;
	}
	float top = -1, top_next = -1;
	int maxm = 0;
	for (auto t : ma)
	{
		if (t.second > maxm)
		{
			maxm = t.second;
			top_next = top;
			top = t.first;
		}
	}
	std::vector<cv::Vec4f> out_lines;
	for (int i = 0; i < vecLines.size(); i++)
	{
		if (abs(lines_k[i]) - 1e-3 <= top || abs(lines_k[i]) - 1e-3 <= top_next)
			out_lines.push_back(vecLines[i]);
	}
	lsd->drawSegments(img, out_lines);
	cv::imshow("reslineMat", img);
	cv::waitKey();
}

#ifdef USE_FABMAP	
	void test_fabmap_op()
	{
		std::string file = "D:\\BaiduNetdiskDownload\\LSD_room_images\\LSD_room\\images\\00001.png";
		cv::Mat image_raw = cv::imread(file, CV_LOAD_IMAGE_GRAYSCALE);
		lsd_slam::c_FabMap* fabmap = new lsd_slam::c_FabMap();
		int a, b;
		fabmap->compareAndAdd(image_raw, &a, &b);
		std::cout << a << ' ' << b << std::endl;
	}
#endif

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

	int show_img_id = 0;
void show_image(
		cv::Mat img1,
		cv::Mat img2,
		std::vector<cv::KeyPoint>& point_k1,
		std::vector<cv::KeyPoint>& point_k2,
		std::vector<cv::line_descriptor::KeyLine>& line_k1,
	std::vector<cv::line_descriptor::KeyLine>& line_k2
)
{
	int width = img1.rows;
	int height = img1.cols;
	cv::Mat res = cv::Mat(width, img1.cols + img2.cols, CV_8UC3);
	cv::Mat roi1(res, cv::Rect(0, 0, img1.cols, img1.rows));
	cv::Mat roi2(res, cv::Rect(img1.cols, 0, img2.cols, img2.rows));
	cv::drawKeypoints(img1, point_k1, roi1, (0, 0, 255));
	cv::drawKeypoints(img2, point_k2, roi2, (0, 0, 255));
	cv::line_descriptor::drawKeylines(img1, line_k1, roi1, (0, 0, 255));
	cv::line_descriptor::drawKeylines(img2, line_k2, roi2, (0,0 , 255));
	
	for (int i = 0; i < point_k1.size(); i++)
	{
		cv::line(res, point_k1[i].pt, cv::Point(point_k2[i].pt.x + img1.cols, point_k2[i].pt.y),cv::Scalar(0,255,0));
	}
	
	for (int i = 0; i < line_k1.size(); i++)
	{
		cv::line(res, line_k1[i].getStartPoint(), line_k1[i].getEndPoint(), cv::Scalar(0, 0, 255));
		cv::line(res, cv::Point(line_k2[i].startPointX + img1.cols, line_k2[i].startPointY), cv::Point(line_k2[i].endPointX + img1.cols, line_k2[i].endPointY), cv::Scalar(0, 0, 255));
		cv::line(res, cv::Point(line_k1[i].startPointX, line_k1[i].startPointY), cv::Point(line_k2[i].startPointX + img1.cols, line_k2[i].startPointY), cv::Scalar(0, 255, 0));
	}

	std::string out_img = "D:\\slam_all\\init\\result\\c_PL_AdaLAM" + std::to_string(show_img_id)+".png";
	cv::imwrite(out_img.c_str(), res);
	//cv::imshow("test", res);
	//cv::waitKey();
	show_img_id++;
}

void get_line_and_point_frame(
	int id,
	cv::Mat img1,cv::Mat img2,
	std::vector<std::vector<cv::KeyPoint> >& frame_kp,
	std::vector<std::vector<cv::line_descriptor::KeyLine> >& frame_kl,
	std::vector<std::vector<std::vector<int>>>& frame_point_map,
	std::vector<std::vector<std::vector<int>>>& frame_line_map)
{
	auto t_start = std::chrono::steady_clock::now();
	cv::Ptr<cv::line_descriptor::BinaryDescriptor> lbd = cv::line_descriptor::BinaryDescriptor::createBinaryDescriptor();
	std::vector<cv::line_descriptor::KeyLine> line_k1, line_k2;
	cv::Ptr<cv::line_descriptor::LSDDetector> lsd = cv::line_descriptor::LSDDetector::createLSDDetector();
	lsd->detect(img1, line_k1, 1, 1);
	lsd->detect(img2, line_k2, 1, 1);
	cv::Mat line_d1, line_d2;//339*32

	/*
	line_d1.zeros(0, 0,CV_32F);
	line_d2.zeros(0, 0, CV_32F);
	line_k1.clear();
	line_k2.clear();
	*/

	lbd->compute(img1, line_k1, line_d1);
	lbd->compute(img2, line_k2, line_d2);

	cv::Ptr<cv::ORB> orb = cv::ORB::create(1000);

	std::vector<cv::KeyPoint> point_k1, point_k2;
	orb->detect(img1, point_k1);
	orb->detect(img2, point_k2);
	cv::Mat point_d1, point_d2;//500*32
	orb->compute(img1, point_k1, point_d1);
	orb->compute(img2, point_k2, point_d2);

	auto t_end = std::chrono::steady_clock::now();
	double dr_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
	std::cout << dr_ms << std::endl;

	int width = img1.rows;
	int height = img1.cols;
	std::vector<int> point1_result, point2_result, line1_result, line2_result;
	c_PL_AdaLAM m_c_PL_AdaLAM = c_PL_AdaLAM(
		width, height,
		point_d1, point_d2,
		point_k1,
		point_k2,
		line_d1, line_d2,
		line_k1,
		line_k2);

	m_c_PL_AdaLAM.core(point1_result, point2_result, line1_result, line2_result);
	int n_point = point1_result.size();
	int n_line = line1_result.size();


	std::vector<std::vector<int>> tmp_point_map(n_point);
	std::vector<std::vector<int>> tmp_line_map(n_line);

	std::vector<cv::KeyPoint> point_res_k1(n_point), point_res_k2(n_point);
	for (int i = 0; i < point1_result.size(); i++)
	{
		point_res_k1[i] = point_k1[point1_result[i]];
		point_res_k2[i] = point_k2[point2_result[i]];
		std::vector<int> tmp(2);
		tmp[0] = point1_result[i];
		tmp[1] = point2_result[i];
		tmp_point_map[i] = tmp;
	}
	std::cout << point_res_k1.size() << std::endl;
	std::vector<cv::line_descriptor::KeyLine> line_res_k1(n_line), line_res_k2(n_line);
	for (int i = 0; i < line1_result.size(); i++)
	{
		line_res_k1[i] = line_k1[line1_result[i]];
		line_res_k2[i] = line_k2[line2_result[i]];
		std::vector<int> tmp(2);
		tmp[0] = line1_result[i];
		tmp[1] = line2_result[i];
		tmp_line_map[i] = tmp;
	}
	show_image(img1, img2, point_res_k1, point_res_k2, line_res_k1, line_res_k2);

	if (id == 1)
	{
		frame_kp[0] = point_k1;
		frame_kl[0] = line_k1;
	}

	frame_kp[id] = point_k2;
	frame_kl[id] = line_k2;

	frame_point_map[id-1] = tmp_point_map;
	frame_line_map[id-1] = tmp_line_map;
}

void decompose_H(
	Eigen::Matrix3d& H21, Eigen::Matrix3d& K,
	std::vector<Eigen::Matrix3f>& res_R,
	std::vector<Eigen::Vector3f>& res_t)
{
	Eigen::Matrix3d invK = K.inverse();
	Eigen::Matrix3d A = invK * H21 * K;

	Eigen::JacobiSVD<Eigen::MatrixXd> svd(A,Eigen::ComputeThinU | Eigen::ComputeThinV);
	Eigen::Matrix3d V = svd.matrixV(), U = svd.matrixU();
	Eigen::Matrix3d Vt = V.transpose();
	Eigen::Matrix3d  w = U.inverse() * A * V.transpose().inverse();

	float s = U.determinant() * Vt.determinant();

	float d1 = w(0,0);
	float d2 = w(1,1);
	float d3 = w(2,2);

	std::vector<Eigen::Matrix3f> vR;
	std::vector<Eigen::Vector3f>vt, vn;
	vR.reserve(8);
	vt.reserve(8);
	vn.reserve(8);

	// step2�����㷨����
	// n'=[x1 0 x3] 4 posibilities e1=e3=1, e1=1 e3=-1, e1=-1 e3=1, e1=e3=-1
	// ������n'= [x1 0 x3] ��Ӧppt�Ĺ�ʽ17
	float aux1 = sqrt((d1 * d1 - d2 * d2) / (d1 * d1 - d3 * d3));
	float aux3 = sqrt((d2 * d2 - d3 * d3) / (d1 * d1 - d3 * d3));
	float x1[] = { aux1,aux1,-aux1,-aux1 };
	float x3[] = { aux3,-aux3,aux3,-aux3 };

	// step3���ָ���ת����
	// step3.1������ sin(theta)��cos(theta)��case d'=d2
	// ����ppt�й�ʽ19
	float aux_stheta = sqrt((d1 * d1 - d2 * d2) * (d2 * d2 - d3 * d3)) / ((d1 + d3) * d2);

	float ctheta = (d2 * d2 + d1 * d3) / ((d1 + d3) * d2);
	float stheta[] = { aux_stheta, -aux_stheta, -aux_stheta, aux_stheta };

	// step3.2������������ת����R��t
	// ������ת���� R��������ppt�й�ʽ18
	//      | ctheta      0   -aux_stheta|       | aux1|
	// Rp = |    0        1       0      |  tp = |  0  |
	//      | aux_stheta  0    ctheta    |       |-aux3|

	//      | ctheta      0    aux_stheta|       | aux1|
	// Rp = |    0        1       0      |  tp = |  0  |
	//      |-aux_stheta  0    ctheta    |       | aux3|

	//      | ctheta      0    aux_stheta|       |-aux1|
	// Rp = |    0        1       0      |  tp = |  0  |
	//      |-aux_stheta  0    ctheta    |       |-aux3|

	//      | ctheta      0   -aux_stheta|       |-aux1|
	// Rp = |    0        1       0      |  tp = |  0  |
	//      | aux_stheta  0    ctheta    |       | aux3|
	for (int i = 0; i < 4; i++)
	{
		Eigen::Matrix3d Rp = Eigen::Matrix3d::Identity(3, 3);
		Rp(0, 0) = ctheta;
		Rp(0, 2) = -stheta[i];
		Rp(2, 0) = stheta[i];
		Rp(2, 2) = ctheta;

		Eigen::Matrix3d R = s * U * Rp * Vt;
		vR.push_back(R.cast<float>());

		Eigen::Vector3d tp;
		tp[0] = x1[i];
		tp[1] = 0;
		tp[2] = -x3[i];
		tp *= d1 - d3;

		// ������Ȼ��t�й�һ������û�о�����Ŀ����SLAM���̵ĳ߶�
		// ��ΪCreateInitialMapMonocular������3D����Ȼ����ţ�Ȼ�󷴹����� t �иı�
		Eigen::Vector3d t = U * tp;
		vt.push_back((t / t.norm()).cast<float>());

		Eigen::Vector3d np;
		np[0] = x1[i];
		np[1] = 0;
		np[2] = x3[i];

		Eigen::Vector3d n = V * np;
		if (n[2] < 0)
			n = -n;
		vn.push_back(n.cast<float>());
	}

	// step3.3������ sin(theta)��cos(theta)��case d'=-d2
	// ����ppt�й�ʽ22
	float aux_sphi = sqrt((d1 * d1 - d2 * d2) * (d2 * d2 - d3 * d3)) / ((d1 - d3) * d2);

	float cphi = (d1 * d3 - d2 * d2) / ((d1 - d3) * d2);
	float sphi[] = { aux_sphi, -aux_sphi, -aux_sphi, aux_sphi };

	// step3.4������������ת����R��t
	// ������ת���� R��������ppt�й�ʽ21
	for (int i = 0; i < 4; i++)
	{
		Eigen::Matrix3d Rp = Eigen::Matrix3d::Identity();
		Rp(0, 0) = cphi;
		Rp(0, 2) = sphi[i];
		Rp(1, 1) = -1;
		Rp(2, 0) = sphi[i];
		Rp(2, 2) = -cphi;

		Eigen::Matrix3d R = s * U * Rp * Vt;
		vR.push_back(R.cast<float>());

		Eigen::Vector3d tp;
		tp[0] = x1[i];
		tp[1] = 0;
		tp[2] = x3[i];
		tp *= d1 + d3;

		Eigen::Vector3d t = U * tp;
		vt.push_back((t / t.norm()).cast<float>());

		Eigen::Vector3d np;
		np[0] = x1[i];
		np[1] = 0;
		np[2] = x3[i];

		Eigen::Vector3d n = V * np;
		if (n[2] < 0)
			n = -n;
		vn.push_back(n.cast<float>());
	}
	res_R = vR;
	res_t = vt;
}

void compute_P3d(const Eigen::Vector3f& p1,const Eigen::Vector3f& p2,Eigen::Matrix<float,3,4>& P1,Eigen::Matrix<float,3,4>& P2,Eigen::Vector3f& P_res)
{
	Eigen::Matrix4f A;

	A.row(0) = p1[0] * P1.row(2) - P1.row(0);
	A.row(1) = p1[1] * P1.row(2) - P1.row(1);
	A.row(2) = p2[0] * P2.row(2) - P2.row(0);
	A.row(3) = p2[1] * P2.row(2) - P2.row(1);
	
	Eigen::MatrixXf u, w, v;
	Eigen::JacobiSVD<Eigen::MatrixXf> svd(A, Eigen::ComputeThinV);
	v = svd.matrixV();
	P_res<<v(0,3),v(1,3),v(2,3);
}

void initialize_points(std::vector<world_point>& points, std::vector<std::vector<cv::KeyPoint> >& frame_kp,std::vector<Sophus::SE3f>& frame_pose,Eigen::Matrix3d& K)
{
	int n = points.size();
	Sophus::SE3f pose = frame_pose[1];
	Eigen::Matrix3f R = pose.inverse().rotationMatrix();
	Eigen::Vector3f t = pose.inverse().translation();
	Eigen::Matrix<float, 3, 4>P1, P2;
	P1 = Eigen::Matrix<float, 3, 4>::Zero();

	P1(0, 0) = K(0, 0); P1(1, 1) = K(1, 1); P1(0, 2) = K(0, 2); P1(1, 2) = K(1, 2); P1(2, 2) = K(2, 2);
	for (int ei = 0; ei < 3; ei++)
		for (int ej = 0; ej < 3; ej++)
			P2(ei, ej) = R(ei, ej);
	for (int ei = 0; ei < 3; ei++)
		P2(ei, 3) = t[ei];
	P2 = K.cast<float>() * P2;
	std::vector<cv::KeyPoint> pre_point = frame_kp[0];
	std::vector<cv::KeyPoint> cur_point = frame_kp[1];
	float mean_depth = 0;
	for (int j = 0; j < n; j++)
	{
		int pos1 = points[j].frame_pos[0];
		int pos2 = points[j].frame_pos[1];
		double x1 = pre_point[pos1].pt.x;
		double y1 = pre_point[pos1].pt.y;
		double x2 = cur_point[pos2].pt.x;
		double y2 = cur_point[pos2].pt.y;
		Eigen::Vector3f p1(x1, y1, 1), p2(x2, y2, 1);
		//p1 = K.inverse().cast<float>() * p1;
		//p2 = K.inverse().cast<float>() * p2;

		Eigen::Vector3f P;
		compute_P3d(p1, p2, P1, P2, P);
		points[j].pos = P;
		mean_depth += P[2];
	}
	/*
	mean_depth /= n;
	
	Sophus::SE3f inv_pose = Sophus::SE3f(R, t / mean_depth);
	frame_pose[1] = inv_pose.inverse();
	for (int i = 0; i < n; i++)
	{
		points[i].pos /= mean_depth;
	}
	
	*/
}

void initialize_lines(std::vector<world_line>& lines, std::vector<std::vector<cv::line_descriptor::KeyLine> >& frame_kl, std::vector<Sophus::SE3f>& frame_pose, Eigen::Matrix3d& K)
{
	int n_lines = lines.size();
	Eigen::Matrix3f K_inv = K.inverse().cast<float>();
	Eigen::Matrix3f rot = frame_pose[1].rotationMatrix();
	Eigen::Vector3f trans = frame_pose[1].translation();
	for (int i = 0; i < n_lines; i++)
	{
		int pos1 = lines[i].frame_pos[0];
		int pos2 = lines[i].frame_pos[1];
		cv::line_descriptor::KeyLine src_line = frame_kl[0][pos1];
		cv::line_descriptor::KeyLine dst_line = frame_kl[1][pos2];
		Eigen::Vector3f src_p(src_line.startPointX, src_line.startPointY, 1.0f), src_q(src_line.endPointX, src_line.endPointY, 1.0f);
		Eigen::Vector3f dst_p(dst_line.startPointX, dst_line.startPointY, 1.0f), dst_q(dst_line.endPointX, dst_line.endPointY, 1.0f);
		src_p = K_inv * src_p; src_q = K_inv * src_q; dst_p = K_inv * dst_p; dst_q = K_inv * dst_q;
		Eigen::Vector3f src_n = src_p.cross(src_q);
		Eigen::Vector3f dst_n = dst_p.cross(dst_q);
		float m2 = -dst_n.dot(trans);
		Eigen::Vector3f v = src_n.cross(dst_n);
		float v_norm = v.norm();
		v /= v_norm;
		Eigen::Vector3f n = src_n;
		n.normalize();
		float m_x = -m2 / dst_n[0];
		float m_z = -m_x * src_n[0] * dst_n[1] / (src_n[2]*dst_n[1] - src_n[1] * dst_n[2]);
		float m_y = -m_z * dst_n[2] / dst_n[1];
		Eigen::Vector3f tmp_m(m_x, m_y, m_z);
		float leng = tmp_m.cross(v).norm();
		n = leng * n;
		Eigen::Matrix<float, 6, 1> space_line;
		space_line << n, v;
		Eigen::Matrix<float, 4, 1> orth;
		lines[i].to_orth(space_line);
	}
}

void get_cmp_result()
{
	int key_n = 10;

	
	std::string all_path = "D:\\slam_all\\init\\match\\dataset.txt";
	std::string path = "D:\\download\\hpatches-sequences-release.tar\\hpatches-sequences-release\\";
	std::string out_pl_time_file = "D:\\slam_all\\init\\match\\pl_time.txt";
	std::string out_point_time_file = "D:\\slam_all\\init\\match\\point_time.txt";
	std::string out_pl_acc_file = "D:\\slam_all\\init\\match\\pl_acc.txt";
	std::string out_point_acc_file = "D:\\slam_all\\init\\match\\point_acc.txt";

	FILE* all_path_fp = fopen(all_path.c_str(), "r");
	char str[30];
	std::vector<std::string> paths;
	while (fscanf(all_path_fp, "%s", str) != EOF)
	{
		paths.push_back(std::string(str));
	}
	fclose(all_path_fp);

	//FILE* out_pl_time_fp = fopen(out_pl_time_file.c_str(), "w");
	//FILE* out_pl_acc_fp = fopen(out_pl_acc_file.c_str(), "w");
	FILE* out_point_time_fp = fopen(out_point_time_file.c_str(), "w");
	FILE* out_point_acc_fp = fopen(out_point_acc_file.c_str(), "w");
	int is_resize = 1;
	for (int i = 0; i < paths.size(); i++)
	{
		std::string dataset_name = paths[i];
		std::vector<std::vector<cv::KeyPoint> > frame_kp(key_n + 1);
		std::vector<std::vector<cv::line_descriptor::KeyLine> > frame_kl(key_n + 1);
		std::vector<std::vector<std::vector<int>>> frame_point_map(key_n);
		std::vector<std::vector<std::vector<int>>> frame_line_map(key_n);
		std::vector<Sophus::SE3f> frame_pose;
		std::vector<double> frame_times(key_n + 1);
		std::vector<cv::Mat> m_file_img(key_n+1);
		for (int j = 0; j <= key_n; j++)
		{
			std::string file_name = path + dataset_name + "\\" + std::to_string(j+1) + ".ppm";
			cv::Mat img = cv::imread(file_name);
			if (img.cols > 600)
			{
				cv::Mat tmp_image;
				cv::resize(img, tmp_image, cv::Size(img.cols / 2, img.rows / 2));
				img = tmp_image;
				is_resize = 2;
			}
			m_file_img[j] = img;
		}
#pragma omp parallel for
		for (int j = 1; j <= key_n; j++)
		{
			auto t_start = std::chrono::steady_clock::now();
			get_line_and_point_frame(j, m_file_img[0], m_file_img[j], frame_kp, frame_kl, frame_point_map, frame_line_map);
			auto t_end = std::chrono::steady_clock::now();
			double dr_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
			frame_times[j] = dr_ms;
		}

		std::vector<double> frame_acc(key_n+1);
		for (int j = 1; j <= key_n; j++)
		{
			std::string H_file = path + dataset_name + "\\H_1_" + std::to_string(j + 1);
			Eigen::Matrix3d H;
			FILE* H_fp = fopen(H_file.c_str(), "r");
			for(int k = 0;k < 3;k++)
				for (int l = 0; l < 3; l++)
				{
					char num[5];
					fscanf(H_fp, "%s",&num);
					H(k, l) = atof(num);
				}
			fclose(H_fp);
			auto p_map = frame_point_map[j - 1];
			auto l_map = frame_line_map[j - 1];

			double p_count = 0;
			for (int k = 0; k < p_map.size(); k++)
			{
				int pos1 = p_map[k][0];
				int pos2 = p_map[k][1];
				auto point1 = frame_kp[0][pos1].pt;
				auto point2 = frame_kp[j][pos2].pt;
				Eigen::Vector3d proj = H * Eigen::Vector3d(point1.x*is_resize, point1.y*is_resize, 1.0f);
				proj /= proj[2];
				Eigen::Vector3d res = Eigen::Vector3d(point2.x * is_resize, point2.y*is_resize, 1.0f) - proj;
				double t = res.norm();
				if (t < 5)
					p_count++;
			}
			if (p_count + l_map.size() < 4)
				frame_acc[j] = 0;
			else
				frame_acc[j] = (p_count+l_map.size()) / (p_map.size()+l_map.size());
		}

		frame_kp.clear();
		frame_kl.clear();
		frame_point_map.clear();
		frame_line_map.clear();
		
		double time = 0;
		double acc = 0;
		for (int j = 1; j <= key_n; j++)
		{
			time += frame_times[j];
			acc += frame_acc[j];
		}
		acc /= key_n;
		time /= key_n;

		//fprintf(out_pl_acc_fp, "%s %f\n",dataset_name.c_str(), acc);
		//fprintf(out_pl_time_fp, "%s %d %d %f\n", dataset_name.c_str(), m_file_img[0].cols, m_file_img[0].rows, time);

		fprintf(out_point_acc_fp, "%s %f\n", dataset_name.c_str(), acc);
		fprintf(out_point_time_fp, "%s %d %d %f\n", dataset_name.c_str(), m_file_img[0].cols, m_file_img[0].rows, time);
	}
	//fclose(out_pl_acc_fp);
	//fclose(out_pl_time_fp);

	fclose(out_point_acc_fp);
	fclose(out_point_time_fp);
}


	int c_PointLineBA::do_BA()
	{
		//get_cmp_result();
		int key_n = 20;

		std::string path = "D:\\FFOutput\\0526";
		//path = "D:\\download\\hpatches-sequences-release.tar\\hpatches-sequences-release\\i_bologna";
		std::vector<std::string> files;
		get_filenames(path, files);
		int n_file = files.size();
		//key_n = n_file-1;
		std::vector<int> img_pl_num(key_n);




		/*
		cv::Mat img1_raw = cv::imread("D:\\FFOutput\\0526\\DJI_0526_00695.jpg");
		cv::Mat img2_raw = cv::imread("D:\\FFOutput\\0526\\DJI_0526_00696.jpg");
		cv::Mat img1 = img1_raw, img2 = img2_raw;
		if (img1_raw.cols > 600)
		{
			cv::resize(img1_raw, img1, cv::Size(img1_raw.cols / 2 , img1_raw.rows / 2 ));
			cv::resize(img2_raw, img2, cv::Size(img1_raw.cols / 2 , img1_raw.rows / 2 ));
		}
		get_line_and_point_frame(1, img1, img2, frame_kp, frame_kl, frame_point_map,frame_line_map);
		*/


		//K(0, 0) = 643.1579; K(0, 2) = 480; K(1, 1) = 643.1579; K(1, 2) = 270; K(2, 2) = 1.0;
		//K(0, 0) = 591.1; K(0, 2) = 331; K(1, 1) = 234; K(1, 2) = 270; K(2, 2) = 1.0;


		std::string img_pl_fime = "D:\\FFOutput\\pl.txt";


		for (int i = 0; i <= key_n; i++)
		{
			std::string file1 = files[i * 2 + 500];
			std::cout << file1 << std::endl;
			cv::Mat image_raw = cv::imread(file1);
			cv::Mat tmp_image;
			if (image_raw.cols > 1000)
			{
				cv::resize(image_raw, tmp_image, cv::Size(image_raw.cols / 2, image_raw.rows / 2));
				image_raw = tmp_image;
			}

			m_file_img.push_back(image_raw);
		}
		

		key_n = m_file_img.size() + 1;
		std::vector<std::vector<cv::KeyPoint> > frame_kp(key_n + 1);
		std::vector<std::vector<cv::line_descriptor::KeyLine> > frame_kl(key_n + 1);
		std::vector<std::vector<std::vector<int>>> frame_point_map(key_n);
		std::vector<std::vector<std::vector<int>>> frame_line_map(key_n);
		std::vector<Sophus::SE3f> frame_pose;
		Eigen::Matrix3d K = m_K.cast<double>();

		//get_line_and_point_frame(0, m_file_img[2], m_file_img[5], frame_kp, frame_kl, frame_point_map, frame_line_map);

	//#pragma omp parallel for
		for (int i = 1; i <= key_n; i++)
		{
			//get_line_and_point_frame(i, m_file_img[i - 1], m_file_img[i], frame_kp, frame_kl, frame_point_map, frame_line_map);
			auto t_start = std::chrono::steady_clock::now();
			get_line_and_point_frame(i, m_file_img[0], m_file_img[i], frame_kp, frame_kl, frame_point_map, frame_line_map);
			auto t_end = std::chrono::steady_clock::now();
			double dr_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
			std::cout << dr_ms << std::endl;
			img_pl_num[i - 1] = frame_point_map[i - 1].size() + frame_line_map[i - 1].size();
			if (frame_point_map[i - 1].size() + frame_line_map[i - 1].size() < 4)
			{
				throw ExceptionNestedException;
				std::cerr << frame_point_map[i - 1].size() + frame_line_map[i - 1].size() << " not enough pl\n";
			}
		}
		FILE* img_pl_fp = fopen(img_pl_fime.c_str(), "w");
		for (int i = 0; i < img_pl_num.size(); i++)
		{
			fprintf(img_pl_fp, "%d %d\n", i, img_pl_num[i]);
		}
		fclose(img_pl_fp);



		for (int i = 1; i <= 1; i++)
		{
			std::vector<std::vector<int>> cur_map = frame_point_map[i - 1];
			std::vector<cv::KeyPoint> pre_point = frame_kp[i - 1];
			std::vector<cv::KeyPoint> cur_point = frame_kp[i];

			Eigen::Matrix<double, 8, 8> A = Eigen::Matrix<double, 8, 8>::Zero();
			Eigen::Matrix<double, 8, 1> v = Eigen::Matrix<double, 8, 1>::Zero();
			/*
			for (int j = 0; j < 4; j++)
			{
				int pos1 = cur_map[j][0];
				int pos2 = cur_map[j][1];
				double x1 = pre_point[pos1].pt.x;
				double y1 = pre_point[pos1].pt.y;
				double x2 = cur_point[pos2].pt.x;
				double y2 = cur_point[pos2].pt.y;

				A(j * 2, 0) = x1;
				A(j * 2, 1) = y1;
				A(j * 2, 2) = 1;
				A(j * 2, 6) = -x1 * x2;
				A(j * 2, 7) = -x2 * y1;
				A(j * 2 + 1, 3) = x1;
				A(j * 2 + 1, 4) = y1;
				A(j * 2 + 1, 5) = 1;
				A(j * 2 + 1, 6) = -x1 * y2;
				A(j * 2 + 1, 7) = -y1 * y2;

				v(j * 2, 0) = x2;
				v(j * 2+1,0) = y2;
			}
			Eigen::Matrix<double,8,1> u = A.colPivHouseholderQr().solve(v);
			Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
			for (int j = 0; j < 8; j++)
			{
				H( j / 3, j % 3) = u(j,0);
			}
			H(2, 2) = 1;

			std::vector<Eigen::Matrix3f> vR;
			std::vector<Eigen::Vector3f> vt;

			decompose_H(H, K, vR, vt);
			*/
			std::vector<cv::Point2f> src_point;
			std::vector<cv::Point2f> dst_point;

			for (int j = 0; j < 100; j++)
			{
				int pos1 = cur_map[j][0];
				int pos2 = cur_map[j][1];
				src_point.push_back(pre_point[pos1].pt);
				dst_point.push_back(cur_point[pos2].pt);

			}

			cv::Mat H = cv::findHomography(src_point, dst_point, CV_RANSAC);
			cv::Mat K_cv;
			std::vector<cv::Mat> R_cv, t_cv, n_cv;
			cv::eigen2cv(K, K_cv);
			cv::decomposeHomographyMat(H, K_cv, R_cv, t_cv, n_cv);
			std::vector<Eigen::Matrix3f> vR;
			std::vector<Eigen::Vector3f> vt;
			for (int k = 0; k < R_cv.size(); k++)
			{
				Eigen::Matrix3d R_tmp;
				Eigen::Vector3d t_tmp;
				cv::cv2eigen(R_cv[k], R_tmp); vR.push_back(R_tmp.cast<float>());
				cv::cv2eigen(t_cv[k], t_tmp); vt.push_back(t_tmp.cast<float>());
			}
			int pos = -1;
			int max_cnt = 0;
			float th2 = 100;
			for (int k = 0; k < vR.size(); k++)
			{
				int tmp_cnt = 0;
				float residual = 0;
				Eigen::Matrix<float, 3, 4>P1, P2;
				P1 = Eigen::Matrix<float, 3, 4>::Zero();

				P1(0, 0) = K(0, 0); P1(1, 1) = K(1, 1); P1(0, 2) = K(0, 2); P1(1, 2) = K(1, 2); P1(2, 2) = K(2, 2);
				for (int ei = 0; ei < 3; ei++)
					for (int ej = 0; ej < 3; ej++)
						P2(ei, ej) = vR[k](ei, ej);
				for (int ei = 0; ei < 3; ei++)
					P2(ei, 3) = vt[k][ei];
				P2 = K.cast<float>() * P2;
				for (int j = 0; j < cur_map.size(); j++)
				{
					int pos1 = cur_map[j][0];
					int pos2 = cur_map[j][1];
					double x1 = pre_point[pos1].pt.x;
					double y1 = pre_point[pos1].pt.y;
					double x2 = cur_point[pos2].pt.x;
					double y2 = cur_point[pos2].pt.y;
					Eigen::Vector3f p1(x1, y1, 1), p2(x2, y2, 1);
					//p1 = K.inverse().cast<float>() * p1;
					//p2 = K.inverse().cast<float>() * p2;

					Eigen::Vector3f P;
					compute_P3d(p1, p2, P1, P2, P);
					Eigen::Vector3f o1 = Eigen::Vector3f::Zero();
					Eigen::Vector3f o2 = -vR[k].transpose() * vt[k];
					if (!isfinite(P[0]) || !isfinite(P[1]) || !isfinite(P[2]))
						continue;
					Eigen::Vector3f normal1 = P - o1;
					float dist1 = normal1.norm();
					Eigen::Vector3f normal2 = P - o2;
					float dist2 = normal2.norm();

					float cos_parall = normal1.dot(normal2) / (dist1 * dist2);

					if (P[2] <= 0 && cos_parall < 0.99998)
						continue;
					Eigen::Vector3f P_t = vR[k] * P + vt[k];
					Eigen::Vector3f test = K.cast<float>() * P_t;
					residual = (test[0] / test[2] - x2) * (test[0] / test[2] - x2) + (test[1] / test[2] - y2) * (test[1] / test[2] - y2);
					//std::cout << k << ' ' << j << ' ' << residual << std::endl;

					if (P_t[2] <= 0 && cos_parall < 0.99998)
						continue;
					if (residual > th2)
						continue;
					tmp_cnt++;
				}
				if (tmp_cnt > max_cnt)
				{
					max_cnt = tmp_cnt;
					pos = k;
				}
			}
			Sophus::SE3f pose = Sophus::SE3f(vR[pos], vt[pos]);
			if (i == 1)
			{
				frame_pose.push_back(Sophus::SE3f());
			}
			Sophus::SE3f pre_pose = frame_pose.back();
			Sophus::SE3f cur_pose = pre_pose * pose.inverse();
			frame_pose.push_back(cur_pose);
		}



		std::vector<std::map<int, int> > all_point(key_n);
		std::vector<std::map<int, int>> all_line(key_n);
		for (int i = 0; i < key_n; i++)
		{
			int n_point = frame_point_map[i].size();
			int n_line = frame_line_map[i].size();
			std::map<int, int> cur_map;
			for (int j = 0; j < n_point; j++)
			{
				cur_map[frame_point_map[i][j][0]] = frame_point_map[i][j][1];
			}
			all_point[i] = cur_map;
			std::map<int, int> cur_line_map;
			for (int j = 0; j < n_line; j++)
			{
				cur_line_map[frame_line_map[i][j][0]] = frame_line_map[i][j][1];
			}
			all_line[i] = cur_line_map;
		}
		std::vector<world_point> points;
		std::vector<world_line> lines;
		for (int i = 0; i < frame_point_map[0].size(); i++)
		{
			int point_pos = frame_point_map[0][i][0];
			int frame_id = 0;
			std::vector<int> frame_point_pos; frame_point_pos.push_back(point_pos);
			while (frame_id < key_n && all_point[frame_id].find(point_pos) != all_point[frame_id].end())
			{
				int t_pos = all_point[frame_id][point_pos];
				frame_point_pos.push_back(t_pos);
				frame_id++;
				point_pos = t_pos;
			}

			if (frame_point_pos.size() != key_n + 1)
			{
				continue;
			}


			world_point pw;
			pw.frame_pos = frame_point_pos;
			points.push_back(pw);
		}

		for (int i = 0; i < frame_line_map[0].size(); i++)
		{
			int line_pos = frame_line_map[0][i][0];
			int frame_id = 0;
			std::vector<int> frame_line_pos; frame_line_pos.push_back(line_pos);
			while (frame_id < key_n && all_line[frame_id].find(line_pos) != all_line[frame_id].end())
			{
				int t_line = all_line[frame_id][line_pos];
				frame_line_pos.push_back(t_line);
				frame_id++;
				line_pos = t_line;
			}

			if (frame_line_pos.size() != key_n + 1)
				continue;

			world_line pl;
			pl.frame_pos = frame_line_pos;
			lines.push_back(pl);
		}

		/*
		std::vector<cv::KeyPoint> src_kp;
		for (int j = 0; j < points.size(); j++)
		{
			int pos = points[j].frame_pos[0];
			src_kp.push_back(frame_kp[0][pos]);
		}
		std::vector<cv::line_descriptor::KeyLine> src_kl;
		for (int j = 0; j < lines.size(); j++)
		{
			int pos = lines[j].frame_pos[0];
			src_kl.push_back(frame_kl[0][pos]);
		}
		for (int i = 1; i <= key_n; i++)
		{
			std::vector<cv::line_descriptor::KeyLine> dst_kl;

			std::vector<cv::KeyPoint> dst_kp;

			for (int j = 0; j < points.size(); j++)
			{
				int pos = points[j].frame_pos[i];
				dst_kp.push_back(frame_kp[i][pos]);
			}
			for (int j = 0; j < lines.size(); j++)
			{
				int pos = lines[j].frame_pos[i];
				dst_kl.push_back(frame_kl[i][pos]);
			}

			show_image(m_file_img[0], m_file_img[i], src_kp, dst_kp, src_kl, dst_kl);
		}
		*/


		initialize_points(points, frame_kp, frame_pose, K);
		initialize_lines(lines, frame_kl, frame_pose, K);

		cv::Mat out_depth(m_file_img[0].rows, m_file_img[0].cols, CV_16UC1);
		cv::Mat out_origin_depth = m_file_img[0].clone();
		std::vector<cv::KeyPoint> out_origin_points;
		for (int i = 0; i < points.size(); i++)
		{
			int pos = points[i].frame_pos[0];
			out_depth.at<ushort>(frame_kp[0][pos].pt) = points[i].pos[2] * 65530;
			out_origin_points.push_back(frame_kp[0][pos]);
		}
		cv::drawKeypoints(out_origin_depth, out_origin_points, out_origin_depth, cv::Scalar(255, 0, 0));
		for (int i = 0; i < lines.size(); i++)
		{
			int pos = lines[i].frame_pos[0];
			float sx = frame_kl[0][pos].startPointX;
			float sy = frame_kl[0][pos].startPointY;
			float ex = frame_kl[0][pos].endPointX;
			float ey = frame_kl[0][pos].endPointY;
			float ix = ex - sx, iy = ey - sy;
			float depth = fabs(cos(lines[i].pos[3]) / sin(lines[i].pos[3]));
			if (sx > 300)
				cv::line(out_depth, frame_kl[0][pos].getStartPoint(), frame_kl[0][pos].getEndPoint(), cv::Scalar(65530 - depth * 65530));
			else
				cv::line(out_depth, frame_kl[0][pos].getStartPoint(), frame_kl[0][pos].getEndPoint(), cv::Scalar(10000));
			cv::line(out_origin_depth, frame_kl[0][pos].getStartPoint(), frame_kl[0][pos].getEndPoint(), cv::Scalar(0, 0, 255));
		}
		cv::imwrite("D:\\slam_all\\init\\depth.png", out_depth);
		cv::imwrite("D:\\slam_all\\init\\origin_depth.png", out_origin_depth);


		std::cout << "points:" << points.size() << std::endl;

		for (int i = 2; i <= key_n; i++)
		{
			std::vector<cv::Point2d> image_point;
			std::vector<cv::Point3d> world_p;
			for (int j = 0; j < points.size(); j++)
			{
				if (points[j].frame_pos.size() > i)
				{
					int image_pos = points[j].frame_pos[i];
					image_point.push_back(frame_kp[i][image_pos].pt);
					cv::Point3d pp(points[j].pos[0], points[j].pos[1], points[j].pos[2]);
					world_p.push_back(pp);
				}
			}
			cv::Mat K_cv;
			Eigen::Matrix3f K_t = K.cast<float>();
			cv::eigen2cv(K, K_cv);


			Sophus::SE3f pre_pose = frame_pose[i - 1];
			Eigen::Matrix3d R_pre = pre_pose.rotationMatrix().transpose().cast<double>();
			Eigen::Vector3d t_pre = -pre_pose.translation().cast<double>();
			cv::Mat r_cv, R_cv, t_cv;
			cv::eigen2cv(t_pre, t_cv);
			cv::eigen2cv(R_pre, R_cv);
			cv::Rodrigues(R_cv, r_cv);

			cv::solvePnP(world_p, image_point, K_cv, cv::Mat(), r_cv, t_cv, true, cv::SOLVEPNP_ITERATIVE);
			cv::Rodrigues(r_cv, R_cv);
			Eigen::Matrix3d R_eigen;
			Eigen::Vector3d t_eigen;
			cv::cv2eigen(R_cv, R_eigen);
			cv::cv2eigen(t_cv, t_eigen);
			Sophus::SE3f pose = Sophus::SE3d(R_eigen, t_eigen).cast<float>();
			frame_pose.push_back(pose.inverse());
		}

		m_K = K.cast<float>();
		std::string dic_files = "D:\\slam_all\\init\\dic.txt";
		FILE* fp = fopen(dic_files.c_str(), "w+");
		for (int i = 0; i < files.size(); i++)
		{
			std::string str = files[i];
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

		SolveBA(K, points, frame_kp, frame_pose);

		std::string filename = "D:\\slam_all\\init\\my_pose.txt";
		fp = fopen(filename.c_str(), "w+");
		for (int i = 0; i < frame_pose.size(); i++)
		{
			Sophus::SE3f pose = frame_pose[i];
			Eigen::Vector3f trans = pose.translation();
			Eigen::Quaternionf rot = pose.unit_quaternion();

			fprintf(fp, "%d ", i);
			for (int j = 0; j < 3; j++)
				fprintf(fp, "%f ", trans[j]);
			fprintf(fp, "%f %f %f %f", rot.x(), rot.y(), rot.z(), rot.w());
			fprintf(fp, "\n");
		}
		fclose(fp);




	}
