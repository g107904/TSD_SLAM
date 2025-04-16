#pragma once
#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <map>
#include <opencv2/line_descriptor.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core/eigen.hpp>
#include <cmath>
#include <Eigen/LU>
#include <Eigen/Dense>
#include <omp.h>


struct d_and_pos {
	float dist;
	int pos;
	bool operator < (const d_and_pos& b)
	{
		return this->dist < b.dist;
	}
};

struct line_des {
	Eigen::Vector2f m_p1, m_q1;
	int m_pos;
	Eigen::Vector3f m_I;
	Eigen::Vector2f m_M;
	Eigen::Vector2f m_T;
	line_des()
	{}
	line_des(cv::line_descriptor::KeyLine& p,int pos)
	{
		m_p1 = Eigen::Vector2f(p.startPointX,p.startPointY);
		m_q1 = Eigen::Vector2f(p.endPointX,p.endPointY);
		m_M = update_Mx(m_p1,m_q1);
		m_T = update_Tx(m_p1, m_q1);
		m_I = update_Ix(m_p1, m_q1);
		this->m_pos = pos;
	}

	static Eigen::Vector2f update_Mx(Eigen::Vector2f p,Eigen::Vector2f q)
	{
		return (p + q) / 2;
	}

	static Eigen::Vector2f update_Tx(Eigen::Vector2f p, Eigen::Vector2f q)
	{
		Eigen::Vector2f T = q - p;
		float dis = T.norm();
		T = T / dis;
		return T;
	}

	static Eigen::Vector3f update_Ix(Eigen::Vector2f p, Eigen::Vector2f q)
	{
		Eigen::Vector3f I;
		I[2] = p[0] * q[1] - q[0] * p[1];
		I[0] = p[1] - q[1];
		I[1] = q[0] - p[0];
		float dis = I.norm();
		I = I / dis;
		return I;
	}

	float point_in_line(float px,float py)
	{
		Eigen::Vector3f p(px, py, 1.0f);
		return fabs(m_I.dot(p));
	}

	float point_dist_mid(cv::KeyPoint& b)
	{
		float x = b.pt.x - m_M[0];
		float y = b.pt.y - m_M[1];
		return (x * x + y * y);
	}

	bool line_point_dist(cv::KeyPoint& p, float threshold)
	{
		Eigen::Vector2f px(p.pt.x, p.pt.y);
		float dist1 = (m_p1 - px).norm();
		float dist2 = (m_q1 - px).norm();
		float dist = (m_M - px).norm();
		//return dist < threshold;
		return dist1 < threshold&& dist2 < threshold;
	}
};


class c_PL_AdaLAM
{
	int m_area_ratio = 100;
	int m_search_expansion = 4;
	int m_ransac_iters = 128;
	int m_min_inliers = 6;
	int m_min_confidence = 200;
	int m_orientation_difference_threshold = 30;
	float m_scale_rate_threshold = 1.5;
	int m_detect_scale_rate_threshold = 5;
	bool m_refit = true;
	bool m_force_seed_mnn = true;
	int m_width, m_height;
	Eigen::MatrixXf m_point_d1, m_point_d2;
	Eigen::MatrixXf m_line_d1, m_line_d2;
	std::vector<cv::KeyPoint> m_point_k1, m_point_k2;
	std::vector<cv::line_descriptor::KeyLine> m_line_k1, m_line_k2;
	std::vector<line_des> m_line_des1,m_line_des2;
	std::vector<std::vector<d_and_pos> > m_point_dist1, m_point_dist2;
	std::vector<int> m_fnn12_point, m_fnn12_line;
	std::vector<float> m_point_score, m_line_score;
public:
	c_PL_AdaLAM(
		int width,int height,
		cv::Mat& point_d1, cv::Mat& point_d2,
		std::vector<cv::KeyPoint>& point_k1,
		std::vector<cv::KeyPoint>& point_k2,
		cv::Mat& line_d1, cv::Mat& line_d2,
		std::vector<cv::line_descriptor::KeyLine>& line_k1,
		std::vector<cv::line_descriptor::KeyLine>& line_k2)
	{
		m_width = width;
		m_height = height;
		cv::cv2eigen(point_d1, m_point_d1);
		cv::cv2eigen(point_d2, m_point_d2);
		if(line_k1.size() > 0)
			cv::cv2eigen(line_d1, m_line_d1);
		if(line_k2.size() > 0)
			cv::cv2eigen(line_d2, m_line_d2);
		m_point_k1 = point_k1;
		m_point_k2 = point_k2;
		m_line_k1 = line_k1;
		m_line_k2 = line_k2;
		int n = line_k1.size();
		int n2 = line_k2.size();
		m_line_des1.resize(n);
		m_line_des2.resize(n2);
		for (int i = 0; i < n; i++)
		{
			m_line_des1[i] = line_des(line_k1[i], i);
			
		}
		for(int i = 0;i < n2;i++)
			m_line_des2[i] = line_des(line_k2[i], i);
		m_point_score.resize(point_k1.size());
		m_line_score.resize(line_k1.size());
		m_fnn12_point.resize(point_k1.size());
		m_fnn12_line.resize(line_k1.size());

	}

	void dist_matrix(Eigen::MatrixXf& d1, Eigen::MatrixXf& d2, std::vector<std::vector<d_and_pos> >& out)
	{
		int n = d1.rows();
		int c = d1.cols();
		int n2 = d2.rows();
		std::vector<d_and_pos> tmp(n2);
		
		Eigen::MatrixXf dTd = d1 * d2.transpose();

		
		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < n2; j++)
			{
				tmp[j].dist = -2 * dTd(i, j);
				tmp[j].pos = j;
			}
			out.push_back(tmp);
		}
		


		std::vector<float> tmp_d1(n,0);
		std::vector<float> tmp_d2(n2,0);
#pragma omp parallel for
		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < c; j++)
			{
				tmp_d1[i] += d1(i, j) * d1(i, j);
			}
		}
#pragma omp parallel for
		for (int i = 0; i < n2; i++)
		{
			for (int j = 0; j < c; j++)
			{
				tmp_d2[i] += d2(i, j) * d2(i, j);
			}
		}
#pragma omp parallel for
		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < n2; j++)
			{
				out[i][j].dist += tmp_d1[i] + tmp_d2[j];
			}
		}

		dTd.resize(0, 0);
		tmp.clear();
		
	}

	void match_and_filter(
		Eigen::MatrixXf& d1, Eigen::MatrixXf& d2,
		std::vector<float>& scores, std::vector<int>& fnn12,
		std::vector<int>& candidate)
	{
		std::vector<std::vector<d_and_pos> > distmat;
		dist_matrix(d1, d2, distmat);

		int n = m_point_d1.rows();

#pragma omp parallel for
		for (int i = 0; i < n; i++)
		{
			sort(distmat[i].begin(), distmat[i].end());
			fnn12[i] = distmat[i][0].pos;
			scores[i] = distmat[i][0].dist / distmat[i][1].dist;
			if (scores[i] < 1e-3)
				scores[i] = 1e-3;
		}

		std::vector<std::vector<d_and_pos> > tmp_d2_to_d1;
		dist_matrix(d2, d1, tmp_d2_to_d1);

		int n2 = m_point_d2.rows();
		for (int i = 0; i < n2; i++)
		{
			sort(tmp_d2_to_d1[i].begin(), tmp_d2_to_d1[i].end());
			int d1_pos = tmp_d2_to_d1[i][0].pos;
			if (fnn12[d1_pos] == i)
			{
				candidate.push_back(d1_pos);
			}
		}
	}

	void calcute_line_score(
		Eigen::MatrixXf& d1, Eigen::MatrixXf& d2,
		std::vector<float>& scores, std::vector<int>& fnn12)
	{
		std::vector<std::vector<d_and_pos> > distmat;
		dist_matrix(d1, d2, distmat);

		int n = d1.rows();

#pragma omp parallel for
		for (int i = 0; i < n; i++)
		{
			sort(distmat[i].begin(), distmat[i].end());
			fnn12[i] = distmat[i][0].pos;
			scores[i] = distmat[i][0].dist / distmat[i][1].dist;
			if (scores[i] < 1e-3)
				scores[i] = 1e-3;
		}
	}

	void compute_point_dist(
		std::vector<cv::KeyPoint>& k1,
		std::vector<std::vector<d_and_pos> >& out
	)
	{
		int n = k1.size();
		Eigen::MatrixXf tmp(n, 2);
#pragma omp parallel for
		for (int i = 0; i < n; i++)
		{
			tmp(i, 0) = k1[i].pt.x;
			tmp(i, 1) = k1[i].pt.y;
		}
		dist_matrix(tmp, tmp, out);
	}

	void select_seeds(float R1,std::vector<int>& candidate,
		std::vector<float>& scores, std::vector<int>& fnn12,
		std::vector<int>& im1seeds, std::vector<int>& im2seeds)
	{
		

		int n = candidate.size();
		for (int i = 0; i < n; i++)
		{
			bool is_minm = true;
			int pos = candidate[i];
			if (scores[pos] >= 0.8 * 0.8)
				continue;
			for (int j = 0; j < n; j++)
			{
				if (j == i)
					continue;
				int can_pos = candidate[j];
				if (m_point_dist1[pos][can_pos].dist < R1 * R1 && scores[can_pos] < scores[pos])
				{
					is_minm = false;
					break;
				}
			}
			if (is_minm)
			{
				im1seeds.push_back(pos);
				im2seeds.push_back(fnn12[pos]);
			}
		}
	}

	float cut_angle(float angle)
	{
		if (angle < -180)
			angle += 360;
		if (angle >= 180)
			angle -= 360;
		return angle;
	}

	void extract_neighborhood_sets(
		std::vector<std::vector<d_and_pos> >& point_dist_d1, std::vector<std::vector<d_and_pos> >& point_dist_d2,
		std::vector<int>& im1seeds, std::vector<int>& im2seeds,
		float R1, float R2,
		std::vector<int>& fnn12_point,
		std::vector<int>& fnn12_line,
		std::vector<std::vector<int> >& point_neighborhood,
		std::vector<std::vector<int>>& line_neighborhood
	)
	{
		int m = im1seeds.size();
		int n_point = fnn12_point.size();
		int n_line = fnn12_line.size();
		float search_R1 = (m_search_expansion * R1) * (m_search_expansion * R1);
		float search_R2 = (m_search_expansion * R2) * (m_search_expansion * R2);

		for (int i = 0; i < m; i++)
		{
			std::vector<int> tmp_point;
			std::vector<int> tmp_line;
			int pos1 = im1seeds[i];
			int pos2 = im2seeds[i];
			float angle = cut_angle(m_point_k1[pos1].angle-m_point_k2[pos2].angle);
			float octave = m_point_k1[pos1].octave / (m_point_k2[pos2].octave+1e-9);
#pragma omp parallel for
			for (int j = 0; j < n_point; j++)
			{
				int cand_pos1 = j;
				int cand_pos2 = fnn12_point[j];
				float cand_angle = cut_angle(m_point_k1[cand_pos1].angle-m_point_k2[cand_pos2].angle);
				float cand_octave = m_point_k1[cand_pos1].octave / (m_point_k2[cand_pos2].octave+1e-9);
				bool flag = true;
				if (flag && point_dist_d1[pos1][cand_pos1].dist >= search_R1)
					flag = false;
				if (flag && point_dist_d2[pos2][cand_pos2].dist >= search_R2)
					flag = false;
				
				
				if (flag && fabs(cut_angle(angle - cand_angle)) >= m_orientation_difference_threshold)
					flag = false;
				
				
				if (flag && octave > 0.1 && cand_octave > 0.1 && (octave / (cand_octave) >= m_scale_rate_threshold  ||  octave / (cand_octave) <= 1 / m_scale_rate_threshold))
					flag = false;
				

				if (flag)
				{
#pragma omp critical
					{
						tmp_point.push_back(j);
					}
				}
			}
#pragma omp parallel for
			for (int j = 0; j < n_line; j++)
			{
				int cand_pos1 = j;
				int cand_pos2 = fnn12_line[j];
				bool flag = true;
				if (flag &&!m_line_des1[cand_pos1].line_point_dist(m_point_k1[pos1], m_search_expansion * R1 ))
					flag = false;
				if (flag && !m_line_des2[cand_pos2].line_point_dist(m_point_k2[pos2], m_search_expansion * R2 ))
					flag = false;
				float dist1 = fabs(m_line_des1[cand_pos1].point_in_line(m_point_k1[pos1].pt.x,m_point_k1[pos1].pt.y));
				float dist2 = fabs(m_line_des2[cand_pos2].point_in_line(m_point_k2[pos2].pt.x,m_point_k2[pos2].pt.y));
				if (flag && fabs(dist1 - dist2) > R1)
					flag = false;

				if (flag)
				{
#pragma omp critical
					{
						tmp_line.push_back(j);
					}
				}

			}

			int l = tmp_point.size();
			for(int j = 0;j < l;j++)
				for (int k = j + 1; k < l; k++)
				{
					if (m_point_score[tmp_point[k]] < m_point_score[tmp_point[j]])
					{
						int t = tmp_point[j];
						tmp_point[j] = tmp_point[k];
						tmp_point[k] = t;
					}
				}
			l = tmp_line.size();
			for (int j = 0; j < l; j++)
				for (int k = j + 1; k < l; k++)
				{
					if (m_line_score[tmp_line[k]] < m_line_score[tmp_line[j]])
					{
						int t = tmp_line[j];
						tmp_line[j] = tmp_line[k];
						tmp_line[k] = t;
					}
				}
			

			point_neighborhood.push_back(tmp_point);
			line_neighborhood.push_back(tmp_line);
		}
	}

	int find_rk(std::vector<float>& num,int target)
	{
		int l = 0, r = num.size() - 1;
		int ans = -1;
		while (l <= r)
		{
			int mid = l + (r - l) / 2;
			if (num[mid] > target)
				r = mid - 1;
			else
			{
				ans = mid;
				l = mid + 1;
			}
		}
		return ans;

	}

	void get_first_couple(int its, int maxm,Eigen::MatrixXf& out)
	{
		int max_ex_search = int(sqrt(2 * its + 0.25) - 0.5);
		int residual_search = its - max_ex_search * (max_ex_search + 1) / 2;
		out = Eigen::MatrixXf(its, 2);
		int pos = 0;
		for(int i = 1;i <= max_ex_search;i++)
			for (int j = 0; j < i; j++)
			{
				out(pos, 0) = i % maxm;
				out(pos, 1) = j % maxm;
				pos++;
			}
		for (int i = 0; i < residual_search; i++)
		{
			out(pos, 0) = residual_search % maxm;
			out(pos, 1) = i % maxm;
			pos++;
		}

	}

	void calculate_result(
		float R1,float R2,
		std::vector<int>& im1seeds, std::vector<int>& im2seeds,
		std::vector<std::vector<int> >& point_neighborhood,
		std::vector<std::vector<int>>& line_neighborhood,
		std::set<int>& point_res,
		std::set<int>& line_res,
		std::vector<std::vector<int>>& test_samples,
		std::vector<std::vector<int>>& test_samples_line
	)
	{
		int m = im1seeds.size();
		Eigen::MatrixXf ransac_its;
		std::vector<int> inl_counts(m);
		std::vector<float> inl_confidence(m);
		std::vector<std::vector<int>> inl_samples(m);
		for (int i = 0; i < m; i++)
		{
			std::vector<int> tmp_point;
			std::vector<int> tmp_line;
			int n_point = point_neighborhood[i].size();
			int n_line = line_neighborhood[i].size();
			n_line = 0;
			std::vector<float> tmp_point_res(n_point);
			int pos1 = im1seeds[i];
			int pos2 = im2seeds[i];
			
			if (n_point < m_min_inliers)
				continue;

			get_first_couple(m_ransac_iters, n_point, ransac_its);

			float p1x = m_point_k1[pos1].pt.x;
			float p1y = m_point_k1[pos1].pt.y;
			float p2x = m_point_k2[pos2].pt.x;
			float p2y = m_point_k2[pos2].pt.y;
			Eigen::Vector2f p1(p1x,p1y), p2(p2x,p2y);
		
			Eigen::MatrixXf point1_local(n_point,2),point2_local(n_point,2);

			for (int j = 0; j < n_point; j++)
			{
				int can_pos1 = point_neighborhood[i][j], can_pos2 = m_fnn12_point[can_pos1];
				point1_local(j, 0) = m_point_k1[can_pos1].pt.x;
				point1_local(j, 1) = m_point_k1[can_pos1].pt.y;
				point2_local(j, 0) = m_point_k2[can_pos2].pt.x;
				point2_local(j, 1) = m_point_k2[can_pos2].pt.y;
			}

			point1_local.rowwise() -= p1.transpose();
			point1_local = point1_local / (R1 * m_search_expansion);
			point2_local.rowwise() -= p2.transpose();
			point2_local = point2_local / (R2 * m_search_expansion);

			float highest_rk = 0;
			int max_count = 0;
			std::vector<d_and_pos> max_count_pos;
			float max_confidence = 0;

			for (int j = 0; j < m_ransac_iters; j++)
			{
				Eigen::Matrix2f X, Y;
				int ransac_p1 = ransac_its(j, 0), ransac_p2 = ransac_its(j, 1),fnn_ransac_p1 = ransac_p1,fnn_ransac_p2 = ransac_p2;
				X(0, 0) = point1_local(ransac_p1, 0);
				X(0, 1) = point1_local(ransac_p1, 1);
				X(1, 0) = point1_local(ransac_p2, 0);
				X(1,1) = point1_local(ransac_p2, 1);
				Y(0, 0) = point2_local(fnn_ransac_p1, 0);
				Y(0, 1) = point2_local(fnn_ransac_p1, 1);
				Y(1, 0) = point2_local(fnn_ransac_p2, 0);
				Y(1,1) = point2_local(fnn_ransac_p2, 1);
				Eigen::Matrix2f A = X.inverse()*Y;
				if (isnan(A(0, 0)))
				{
					A = Eigen::Matrix2f::Identity();
				}
				Eigen::MatrixXf y_pred = point1_local * A;
				Eigen::MatrixXf residual = point2_local - y_pred;

				std::vector<d_and_pos> res(n_point);

				for (int k = 0; k < n_point; k++)
				{
					res[k].dist = residual(k, 0) * residual(k, 0) + residual(k, 1) * residual(k, 1);
					res[k].pos = k;
				}
				sort(res.begin(), res.end());
				int cnt = 0;
				int k_pos = -1;
				for (int k = 0; k < n_point; k++)
				{
					if (res[k].dist < 1e-8)
						continue;
					if (k_pos == -1)
						k_pos = k;
					float progress_rate = (float)(k - k_pos + 1) / (float)(n_point - k_pos);
					if (res[k].dist * m_min_confidence >= progress_rate)
						break;
					cnt = cnt+1;
				}
				float confidence = (float)cnt / (float)(n_point - k_pos) / highest_rk;
				if (cnt > max_count || confidence > max_confidence )
				{
					max_count = cnt;
					highest_rk = res[cnt - 1].dist;
					max_count_pos = res;
					max_confidence = confidence;
				}
			}

			inl_counts[i]=(max_count);

			inl_confidence[i] = (max_confidence);

			for (int j = 0; j < max_count; j++)
			{
				tmp_point.push_back(point_neighborhood[i][max_count_pos[j].pos]);
			}
			inl_samples[i] = (tmp_point);
		}

		int max_inl_pos = 0;
		int maxm_inl_samples = 0;
		for(int i = 1;i < m;i++)
			if (inl_samples[i].size() > maxm_inl_samples)
			{
				maxm_inl_samples = inl_samples[i].size();
				max_inl_pos = i;
			}
		for (int j = 0; j < std::min(10, (int)inl_samples[max_inl_pos].size()); j++)
		{
			float confidence = inl_confidence[max_inl_pos];
			float cnt = inl_counts[max_inl_pos];
			if (confidence >= m_min_confidence && cnt * (1 - 1 / confidence) >= m_min_inliers)
				point_res.insert(inl_samples[max_inl_pos][j]);
		}
		for (int i = 0; i < m; i++)
		{
			int n_point = point_neighborhood[i].size();

			float confidence = inl_confidence[i];
			float cnt = inl_counts[i];
			Eigen::Matrix2f X;
			Eigen::Matrix2f Y;
			Eigen::MatrixXf A;
			for (int j = 0; j < inl_samples[i].size(); j++)
			{
				if (j <= 1)
				{
					int pos1 = inl_samples[i][j];
					X(j, 0) = m_point_k1[pos1].pt.x;
					X(j, 1) = m_point_k1[pos1].pt.y;
					int pos2 = m_fnn12_point[pos1];
					Y(j, 0) = m_point_k2[pos2].pt.x;
					Y(j, 1) = m_point_k2[pos2].pt.y;
				}
				if (n_point < m_min_inliers)
				    continue;
				if (confidence >= m_min_confidence && cnt * (1 - 1 / confidence) >= m_min_inliers)
					point_res.insert(inl_samples[i][j]);
			}

			std::vector<int> tmp_lines;
			A = X.inverse() * Y;
			int n_line = line_neighborhood[i].size();
			for (int j = 0; j < n_line; j++)
			{
				int line_pos1 = line_neighborhood[i][j];
				if (line_res.find(j) != line_res.end())
					continue;
				Eigen::Vector2f x;
				Eigen::Vector2f y;
				int line_pos2 = m_fnn12_line[line_pos1];
				int seed_pos1 = im1seeds[i], seed_pos2 = m_fnn12_point[seed_pos1];
				Eigen::Vector2f p1_new, q1_new,p2_new,q2_new;
				Eigen::Vector2f M1, T1, T2;
				Eigen::Vector3f I2;
				p1_new << m_line_des1[line_pos1].m_p1[0] - m_point_k1[seed_pos1].pt.x, m_line_des1[line_pos1].m_p1[1] - m_point_k1[seed_pos1].pt.y;
				q1_new << m_line_des1[line_pos1].m_q1[0] - m_point_k1[seed_pos1].pt.x, m_line_des1[line_pos1].m_q1[1] - m_point_k1[seed_pos1].pt.y;
				p2_new << m_line_des2[line_pos2].m_p1[0] - m_point_k2[seed_pos2].pt.x, m_line_des2[line_pos2].m_p1[1] - m_point_k2[seed_pos2].pt.y;
				q2_new << m_line_des2[line_pos2].m_q1[0] - m_point_k2[seed_pos2].pt.x, m_line_des2[line_pos2].m_q1[1] - m_point_k2[seed_pos2].pt.y;
				p1_new /= m_search_expansion * R1;
				q1_new /= m_search_expansion * R1;
				p2_new /= m_search_expansion * R2;
				q2_new /= m_search_expansion * R2;
				T1 = line_des::update_Tx(p1_new, q1_new);
				T2 = line_des::update_Tx(p2_new, q2_new);
				I2 = line_des::update_Ix(p2_new, q2_new);
				Eigen::Vector2f p1_new_prime = A.transpose() * p1_new;
				Eigen::Vector3f p1_new_prime_v3(p1_new_prime[0],p1_new_prime[1], 1.0f);
				Eigen::Vector2f q1_new_prime = A.transpose() * q1_new;
				Eigen::Vector3f q1_new_prime_v3(q1_new_prime[0], q1_new_prime[1], 1.0f);
				float dist = fabs(I2.dot(p1_new_prime_v3)) + fabs(I2.dot(q1_new_prime_v3));
				dist += ((A.transpose() * T1).normalized() - T2).norm();
				if (dist < 1e-1)
				{
					line_res.insert(line_pos1);
					tmp_lines.push_back(line_pos1);
				}
			}
			test_samples_line.push_back(tmp_lines);
		}
		test_samples = inl_samples;
	}

	void core(std::vector<int>& point1_result,std::vector<int>& point2_result,
		std::vector<int>& line1_result,std::vector<int>& line2_result)
	{
		const float pi = 3.1415926f;
		float R1 = sqrt(m_width * m_height / m_area_ratio / pi);
		float R2 = sqrt(m_width * m_height / m_area_ratio / pi);

		compute_point_dist(m_point_k1, m_point_dist1);
		compute_point_dist(m_point_k2, m_point_dist2);
		std::vector<int> tmp;
		match_and_filter(m_point_d1, m_point_d2, m_point_score, m_fnn12_point,tmp);
		calcute_line_score(m_line_d1, m_line_d2, m_line_score, m_fnn12_line);
		std::vector<int> im1seed, im2seed;
		select_seeds(R1, tmp, m_point_score, m_fnn12_point, im1seed, im2seed);

		std::vector<std::vector<int>> point_neighborhood, line_neighborhood;

		extract_neighborhood_sets(m_point_dist1, m_point_dist2, im1seed, im2seed,
			R1, R2, m_fnn12_point, m_fnn12_line, point_neighborhood, line_neighborhood);
		
		std::set<int> point1_res;
		std::set<int> line1_res;
		std::vector<std::vector<int>> test_samples;
		std::vector<std::vector<int>> test_samples_line;
		calculate_result(
			R1,R2,im1seed, im2seed, point_neighborhood, line_neighborhood, point1_res, line1_res,
			test_samples,test_samples_line);

		/*
		point1_result.clear();
		for (int i = 0; i < m_point_k1.size(); i++)
		{
			point1_result.push_back(i);
		}
		for (int i = 0; i < m_line_k1.size(); i++)
		{
			line1_result.push_back(i);
		}
		*/
		
		/*
		point1_result.push_back(im1seed[0]);
		for (int i = 0; i < point_neighborhood[0].size(); i++)
			point1_result.push_back(point_neighborhood[0][i]);
		line1_result = line_neighborhood[0];
		*/
		/*
		point1_result.assign(test_samples[0].begin(), test_samples[0].end());
		line1_result.assign(test_samples_line[0].begin(), test_samples_line[0].end());
		*/
		
		for (auto t : point1_res)
		{
			if (t < 0 || t >= m_point_k1.size())
				continue;
			point1_result.push_back(t);
			
		}
		for (auto t : line1_res)
		{
			if (t < 0 || t >= m_line_k1.size())
				continue;
			line1_result.push_back(t);
		}


		std::map<int, int> p2_p1;
		for (int i = 0; i < point1_result.size(); i++)
		{
			int pos2 = m_fnn12_point[point1_result[i]];
			if (p2_p1.find(pos2) == p2_p1.end())
			{
				p2_p1[pos2] = point1_result[i];
			}
			else
			{
				float src_score = m_point_score[p2_p1[pos2]];
				float dst_score = m_point_score[point1_result[i]];
				if (dst_score < src_score)
				{
					p2_p1[pos2] = point1_result[i];
				}
			}
		}		
		for (auto it = point1_result.begin(); it != point1_result.end(); )
		{
			int pos2 = m_fnn12_point[*it];
			if (p2_p1[pos2] != *it)
			{
				it = point1_result.erase(it);
			}
			else
				it++;
		}

		std::map<int, int> l2_l1;
		for (int i = 0; i < line1_result.size(); i++)
		{
			int pos2 = m_fnn12_line[line1_result[i]];
			if (l2_l1.find(pos2) == l2_l1.end())
			{
				l2_l1[pos2] = line1_result[i];
			}
			else
			{
				float src_score = m_line_score[l2_l1[pos2]];
				float dst_score = m_line_score[line1_result[i]];
				if (dst_score < src_score)
				{
					l2_l1[pos2] = line1_result[i];
				}
			}
		}
		for (auto it = line1_result.begin(); it != line1_result.end(); )
		{
			int pos2 = m_fnn12_line[*it];
			if (l2_l1[pos2] != *it)
			{
				it = line1_result.erase(it);
			}
			else
				it++;
		}


		for (int i = 0; i < point1_result.size(); i++)
		{
			point2_result.push_back(m_fnn12_point[point1_result[i]]);
		}
		for (int i = 0; i < line1_result.size(); i++)
			line2_result.push_back(m_fnn12_line[line1_result[i]]);
		


	}
};