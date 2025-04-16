#pragma once
#include "GlobalMapping/c_KeyFrameGraph.h"
#include <map>
namespace slam_gl {
	struct point {
		float pos[3];
	};

	class c_keyframe_draw {
	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
		c_keyframe_draw(int width,int height,Eigen::Matrix3f K);
		~c_keyframe_draw();

		void set_from_KF(lsd_slam::c_Frame* keyframe);
		void set_from_pose(Sophus::Sim3 pose,int id);
		//void draw_camera();
		void draw_cam(bool is_current_cam);
		void set_gl_buffer();

		//void get_valid_point_raw(int ymin,int ymax);

		//void flush();

		Eigen::Matrix3f m_K;
		int m_width, m_height;

		int m_id;
		Sophus::Sim3 m_cam_to_world;
		float* m_idepth;
		float* m_idepth_var;

		double m_time;
		int m_total_points, m_displayed_points;

		Eigen::Vector3f m_center;
		int m_vertex_num;

		float m_scaled_threshold, m_abs_threshold, m_scale;
		int m_min_near_support;
		int m_sparsify_factor;

		unsigned int m_gl_VAO_id;
		unsigned int m_gl_buffer_id;
		bool m_gl_buffer_id_valid;
		bool m_gl_buffer_valid;

		unsigned int m_gl_cam_vertex_id;
		unsigned int m_gl_cam_element_id;
		unsigned int m_gl_cam_VAO_id;

		std::mutex m_mutex;
	};


	class c_slam_draw {
	public:
		c_slam_draw(int width, int height, Eigen::Matrix3f K,lsd_slam::c_KeyFrameGraph* graph);
		void do_draw_change();
		//void do_draw_change_row(int ymin, int ymax);
		void do_draw();
		void change_current_cam(Sophus::Sim3 pose, int id);
	private:
		lsd_slam::c_KeyFrameGraph* m_graph_pt;

		/*
		std::vector<Sophus::Sim3> m_poses[2];
		std::vector<float*> m_depths[2];
		float* m_points[2];
		*/
		std::vector<c_keyframe_draw*> m_draw_keyframes;
		std::map<int, c_keyframe_draw*> m_id_to_keyframe;

		c_keyframe_draw* m_current_cam;

		int m_width;
		int m_height;
		Eigen::Matrix3f m_K;

		std::mutex m_change_mutex;
		int m_change_flag;

		int m_change_id;

	};
}