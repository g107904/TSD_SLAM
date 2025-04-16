#pragma once
#include <Eigen/Core>
#include <vector>
#include <cmath>
#include "Sophus/so3.hpp"
class world_point
{
public:
	Eigen::Vector3f pos;
	std::vector<int> frame_pos;
};

class world_line
{
public:
	Eigen::Matrix<float, 4, 1> pos;
	std::vector<int> frame_pos;
	
	Eigen::Matrix<float, 6, 1> to_plucker()
	{
		float w1 = cos(pos[3]);
		float w2 = sin(pos[3]);
		Sophus::SO3f rot(pos[0], pos[1], pos[2]);
		Eigen::Matrix3f rotation = rot.matrix();
		Eigen::Vector3f u1 = rotation.col(0);
		Eigen::Vector3f u2 = rotation.col(1);
		float n_norm = w1 / w2;
		Eigen::Vector3f n = n_norm * u1;
		Eigen::Vector3f v = u2;
		Eigen::Matrix<float, 6, 1> out;
		out << n, v;
		return out;
	}

	void to_orth(Eigen::Matrix<float, 6, 1>& nv)
	{
		Eigen::Matrix<float, 3, 2> tmp;
		Eigen::Vector3f n, v;
		for (int i = 0; i < 3; i++)
		{
			n[i] = nv[i];
			v[i] = nv[i + 3];
		}
		tmp << n, v;
		float n_norm = n.norm();
		float theta = atan(1 / n_norm);
		Eigen::HouseholderQR<Eigen::Matrix<float, 3, 2>> qr;
		qr.compute(tmp);
		Eigen::MatrixXf Q = qr.householderQ();
		Eigen::Matrix3f QQ;
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++)
				QQ(i, j) = Q(i, j);
		Eigen::Vector3f euler_angles = QQ.eulerAngles(0, 1, 2);
		pos << euler_angles, theta;
	}
	
};
