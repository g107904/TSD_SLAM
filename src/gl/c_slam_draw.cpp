#include "c_slam_draw.h"
#include "DataStructures/c_Frame.h"
#include "DataStructures/c_FrameMemory.h"
#include "util/c_index_thread_reduce.h"
#include <thread>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "shader.h"
#include "camera.h"
#include <iostream>
namespace slam_gl {
	struct 
	{
		unsigned int SCR_WIDTH = 800;
		unsigned int SCR_HEIGHT = 600;
		bool is_pause = true;
		glm::vec3 lightPos = glm::vec3(1.2f, 1.0f, 2.0f);
		glm::mat4 model = glm::mat4(1.0f);
		double rad = 0, theta = 0, phi = 0, yangle = 0, xangle = 0;

		Camera camera = Camera(glm::vec3(0.0f, 0.0f, 20.0f));

		std::mutex camera_mutex;
		Eigen::Vector3f camera_change_tmp;

		bool firstMouse = true, is_left = true;
		double lastX = SCR_WIDTH / 2.0f;
		double lastY = SCR_HEIGHT / 2.0f;
		float deltaTime = 0.0f;	// time between current frame and last frame
		float lastFrame = 0.0f;
		float cur_frame = 0.0f;
		int pmode = 0;
		int pos_mode = 0;
		int pos_side = 0;
		float cur_camera_center_x = 0, cur_camera_center_y = 0;
		float cur_camera_center_last_x = SCR_WIDTH / 2.0f, cur_camera_center_last_y = SCR_HEIGHT / 2.0f;
		float pi = 3.1415926;
	}data;

	struct {
		float scaled_depth_var_threshold = 1;
		float abs_depth_var_threshold = 1;
		int min_near_support = 5;
		int cut_firstN_KF = 5;
		int sparsify_factor = 1;
		int num_refreshed_already = 0;
		double last_frame_time = 1e15;
	} gl_threshold;

	void mouse_callback(GLFWwindow* window, double xpos, double ypos)
	{
		if (data.firstMouse)
		{
			return;
		}

		if (data.is_left)
		{
			double dx = -(xpos - data.lastX) / data.SCR_WIDTH * data.pi / 2, dy = -(ypos - data.lastY) / data.SCR_HEIGHT * data.pi / 2;
			data.yangle += dx;
			data.xangle += dy;
			data.yangle -= floor(data.yangle / 2 / data.pi) * 2 * data.pi;
			data.xangle -= floor(data.xangle / 2 / data.pi) * 2 * data.pi;
			//cout << xangle << ' ' << yangle << endl;
			data.lastX = xpos;
			data.lastY = ypos;
		}
		else
		{
			double dx = (xpos - data.lastX) / data.SCR_WIDTH * data.pi / 2, dy = -(ypos - data.lastY) / data.SCR_HEIGHT * data.pi / 2;
			data.phi += dy;
			data.theta += dx;
			double x = data.rad * sin(data.phi) * cos(data.theta), y = data.rad * cos(data.phi), z = data.rad * sin(data.phi) * sin(data.theta);
			//camera.Position = glm::vec3(x, y, z) + camera.center;
			//cout << x << ' ' << y << ' ' << z << endl;
			data.lightPos = glm::vec3(x, y, z) + data.camera.center;
			data.lastX = xpos;
			data.lastY = ypos;

		}
	}

	void scroll_callback(GLFWwindow* window, double xpos, double ypos)
	{
		if (ypos > 0)
		{
			if (data.pos_mode == 0)
				data.camera.ProcessKeyboard(FORWARD, data.deltaTime);
		}
		else
		{
			if (data.pos_mode == 0)
				data.camera.ProcessKeyboard(BACKWARD, data.deltaTime);
		}

	}

	void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
	{
		if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS)
		{
			if (data.firstMouse)
			{
				glfwGetCursorPos(window, &data.lastX, &data.lastY);
				data.firstMouse = false;
				data.is_left = false;
			}
		}
		else if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_RELEASE)
		{
			data.firstMouse = true;
		}
		else if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS)
		{
			if (data.firstMouse)
			{
				glfwGetCursorPos(window, &data.lastX, &data.lastY);
				data.firstMouse = false;
				data.is_left = true;
			}
		}
		else if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE)
		{
			data.firstMouse = true;
		}
		else if (button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_PRESS)
		{
			//pmode = (pmode + 1) % 3;
			data.pos_mode = data.pos_mode ^ 1;
			//is_pause = !is_pause;
		}
	}

	void framebuffer_size_callback(GLFWwindow* window, int width, int height)

	{
		data.SCR_WIDTH = width;
		data.SCR_HEIGHT = height;

		if (width == 0)
		{
			data.SCR_HEIGHT = data.SCR_WIDTH = 10;
		}
		//glViewport(0, 0, width, height);

	}

	c_keyframe_draw::c_keyframe_draw(int width,int height,Eigen::Matrix3f K)
	{
		m_width = width;
		m_height = height;
		m_K = K;

		m_id = 0;
		m_cam_to_world = Sophus::Sim3();
		m_idepth = 0;
		m_idepth_var = 0;

		m_time = 0;
		m_total_points = m_displayed_points = 0;
		
		m_scale = 0;
		m_scaled_threshold = m_abs_threshold = 0;
		m_min_near_support = m_sparsify_factor = 0;

		m_gl_buffer_id_valid = m_gl_buffer_valid = 0;
	}

	c_keyframe_draw::~c_keyframe_draw()
	{
		if (m_gl_buffer_id_valid)
		{
			glDeleteBuffers(1, &m_gl_buffer_id);
			m_gl_buffer_id_valid = false;
		}
		if (m_idepth != 0)
		{
			free(m_idepth);
			free(m_idepth_var);
		}
	}

	void c_keyframe_draw::set_from_KF(lsd_slam::c_Frame* keyframe)
	{
		m_id = keyframe->get_id();
		m_cam_to_world = keyframe->get_scaled_cam_to_world();
		if (m_idepth == 0)
		{
			m_idepth = (float*)malloc(sizeof(float) * m_width * m_height);
			m_idepth_var = (float*)malloc(sizeof(float) * m_width * m_height);
		}
		memcpy(m_idepth, keyframe->get_idepth(), (sizeof(float) * m_width * m_height));
		memcpy(m_idepth_var, keyframe->get_idepth_var(), (sizeof(float) * m_width * m_height));
		m_gl_buffer_valid = false;
	}

	void c_keyframe_draw::set_from_pose(Sophus::Sim3 pose, int id)
	{
		m_id = id;
		m_cam_to_world = pose;
	}

	void c_keyframe_draw::draw_cam(bool is_current_cam)
	{


		float* tmp_vertex_buffer = (float*)malloc(sizeof(float) * 5 * 9);
		unsigned int* tmp_element_buffer = (unsigned int*)malloc(sizeof(unsigned int) * 8 * 2);
		
		memset(tmp_vertex_buffer, 0, sizeof(float) * 5 * 9);


		float fx = m_K(0, 0);
		float fy = m_K(1, 1);
		float cx = m_K(0, 2);
		float cy = m_K(1, 2);

		float x1 = 0.05 * (0 - cx) / fx;
		float x2 = 0.05 * (m_width - 1.0f - cx) / fx;
		float y1 = 0.05 * (0 - cy) / fy;
		float y2 = 0.05 * (m_height - 1.0f - cy) / fy;


		tmp_vertex_buffer[0] = 0.0f;
		tmp_vertex_buffer[1] = 0.0f;
		tmp_vertex_buffer[2] = 0.0f;

		//lu
		tmp_vertex_buffer[9] = x1;
		tmp_vertex_buffer[10] = y1;
		tmp_vertex_buffer[11] = 0.05;

		//ld
		tmp_vertex_buffer[18] = x1;
		tmp_vertex_buffer[19] = y2;
		tmp_vertex_buffer[20] = 0.05;

		//ru
		tmp_vertex_buffer[27] = x2;
		tmp_vertex_buffer[28] = y1;
		tmp_vertex_buffer[29] = 0.05;

		//rd
		tmp_vertex_buffer[36] = x2;
		tmp_vertex_buffer[37] = y2;
		tmp_vertex_buffer[38] = 0.05;

		//color
		for (int i = 0; i < 5; i++)
		{
			if (is_current_cam)
			{
				tmp_vertex_buffer[i * 9 + 6] = 1.0f;//red
				tmp_vertex_buffer[i * 9 + 7] = 0.0f;
				tmp_vertex_buffer[i * 9 + 8] = 0.0f;
			}
			else
			{
				tmp_vertex_buffer[i * 9 + 6] = 0.0f;
				tmp_vertex_buffer[i * 9 + 7] = 0.0f;
				tmp_vertex_buffer[i * 9 + 8] = 1.0f;//blue
			}
		}

		//element

		for (int i = 0; i < 4; i++)
		{
			tmp_element_buffer[i * 2] = 0;
			tmp_element_buffer[i * 2 + 1] = i+1;
		}

		//4~8
		tmp_element_buffer[8] = 1; tmp_element_buffer[9] = 2;
		tmp_element_buffer[10] = 1; tmp_element_buffer[11] = 3;  
		tmp_element_buffer[12] = 2; tmp_element_buffer[13] = 4;
		tmp_element_buffer[14] = 3; tmp_element_buffer[15] = 4;


		m_gl_cam_vertex_id = 0;
		m_gl_cam_element_id = 0;
		m_gl_cam_VAO_id = 0;

		glGenVertexArrays(1, &m_gl_cam_VAO_id);
		glGenBuffers(1, &m_gl_cam_vertex_id);
		glGenBuffers(1, &m_gl_cam_element_id);

		glBindVertexArray(m_gl_cam_VAO_id);
		glBindBuffer(GL_ARRAY_BUFFER, m_gl_cam_vertex_id);
		
		glBufferData(GL_ARRAY_BUFFER, 5 * 9 * sizeof(float), tmp_vertex_buffer, GL_STATIC_DRAW);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_gl_cam_element_id);

		glBufferData(GL_ELEMENT_ARRAY_BUFFER, 8 * 2 * sizeof(unsigned int), tmp_element_buffer, GL_STATIC_DRAW);

		free(tmp_vertex_buffer);
		free(tmp_element_buffer);

	}
	void c_keyframe_draw::set_gl_buffer()
	{
		bool is_still_good = m_scaled_threshold == gl_threshold.scaled_depth_var_threshold
			&& m_abs_threshold == gl_threshold.abs_depth_var_threshold
			&& m_scale * 1.2 > m_cam_to_world.scale()
			&& m_scale < m_cam_to_world.scale() * 1.2
			&& m_min_near_support == gl_threshold.min_near_support
			&& m_sparsify_factor == gl_threshold.sparsify_factor;

		if (m_gl_buffer_valid
			&& (
				is_still_good
				|| gl_threshold.num_refreshed_already > 10))
			return;

		gl_threshold.num_refreshed_already++;

		m_gl_buffer_valid = true;

		if (m_gl_buffer_id_valid)
		{
			glDeleteBuffers(1, &m_gl_buffer_id);
			m_gl_buffer_id_valid = false;
		}

		if (m_idepth == 0)
			return;

		m_scaled_threshold = gl_threshold.scaled_depth_var_threshold;
		m_abs_threshold = gl_threshold.abs_depth_var_threshold;
		m_scale = m_cam_to_world.scale();
		m_min_near_support = gl_threshold.min_near_support;
		m_sparsify_factor = gl_threshold.sparsify_factor;

		Eigen::Matrix3f K_inv = m_K.inverse();
		float fxi = K_inv(0, 0);
		float fyi = K_inv(1, 1);
		float cxi = K_inv(0, 2);
		float cyi = K_inv(1, 2);

		m_vertex_num = 0;
		//point* tmp_points = new point[m_width * m_height];
		float* tmp_points = (float*)malloc(sizeof(float) * m_width * m_height*3);

		m_total_points = m_displayed_points = 0;
		m_center.setZero();
		for(int y = 1;y < m_height - 1;y++)
			for (int x = 1; x < m_width - 1; x++)
			{
				int index = x + y * m_width;
				
				if (m_idepth[index] <= 0)
					continue;
				if (m_idepth[index] >= 1.5)
					continue;
				m_total_points++;
				
				if (m_sparsify_factor > 1
					&& rand() % m_sparsify_factor != 0)
					continue;

				float d = 1 / m_idepth[index];
				float d4 = d * d; d4 *= d4;

				if (m_idepth_var[index] * d4 > m_scaled_threshold)
					continue;

				if (m_idepth_var[index] * d4 * m_scale * m_scale > m_abs_threshold)
					continue;

				if (m_min_near_support > 1)
				{
					int near_support = 0;
					for(int dx = -1;dx < 2;dx++)
						for (int dy = -1; dy < 2; dy++)
						{
							int cur_index = index + dx + dy * m_width;
							if (m_idepth[cur_index] > 0)
							{
								float diff = m_idepth[cur_index] - 1.0f / d;
								if (diff * diff < 2 * m_idepth_var[index])
									near_support++;
							}
						}
					if (near_support < m_min_near_support)
						continue;
				}


				tmp_points[m_vertex_num*3+0] = (x * fxi + cxi) * d;
				tmp_points[m_vertex_num*3+1] = (y * fyi + cyi) * d;
				tmp_points[m_vertex_num*3+2] = d;

				for(int j = 0;j < 3;j++)
					m_center[j] += tmp_points[m_vertex_num*3+j];


				m_vertex_num++;
				m_displayed_points++;
			}

		for (int j = 0; j < 3; j++)
			m_center[j] = m_center[j] / m_vertex_num;


		m_gl_VAO_id = 0;
		m_gl_buffer_id = 0;
		glGenVertexArrays(1, &m_gl_VAO_id);
		glGenBuffers(1, &m_gl_buffer_id);

		glBindVertexArray(m_gl_VAO_id);
		glBindBuffer(GL_ARRAY_BUFFER, m_gl_buffer_id);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float) *3 *  m_vertex_num, tmp_points,GL_STATIC_DRAW);

		m_gl_buffer_valid = true;
		free(tmp_points);
	}

	c_slam_draw::c_slam_draw(int width, int height,Eigen::Matrix3f K, lsd_slam::c_KeyFrameGraph* graph)
	{
		m_width = width;
		m_height = height;
		m_K = K;
		m_graph_pt = graph;
		m_change_flag = 0;
		
		m_current_cam = new c_keyframe_draw(m_width, m_height, m_K);
		//memset(m_points, 0, sizeof(m_points));
	}
	
	void c_slam_draw::do_draw_change()
	{
		std::unique_lock<std::mutex> lock(m_graph_pt->m_new_keyframes_mutex);
		m_graph_pt->m_new_keyframes_created_signal.wait(lock);

		m_graph_pt->m_pose_consistency_mutex.lock_shared();
		m_graph_pt->m_keyframes_all_mutex.lock_shared();
		std::vector<c_keyframe_draw*> tmp_buffer;
		std::vector<lsd_slam::c_FramePose*> tmp_poses;
		for (auto KF : m_graph_pt->m_keyframes_all)
		{
			lsd_slam::c_FramePose* pose = KF->m_pose;
			int id = KF->get_id();
			if (m_id_to_keyframe.find(id) == m_id_to_keyframe.end())
			{
				c_keyframe_draw* new_KF = new c_keyframe_draw(m_width, m_height, m_K);
				new_KF->set_from_KF(KF);
				tmp_buffer.push_back(new_KF);
			}
			else
			{
				tmp_poses.push_back(pose);
			}
		}
		m_graph_pt->m_keyframes_all_mutex.unlock_shared();
		m_graph_pt->m_pose_consistency_mutex.unlock_shared();

		m_change_mutex.lock();
		for (int i = 0; i < tmp_poses.size(); i++)
		{
			int id = tmp_poses[i]->m_frame_ID;
			m_id_to_keyframe[id]->m_cam_to_world = tmp_poses[i]->get_cam_to_world();
		}
		for (int i = 0; i < tmp_buffer.size(); i++)
		{
			m_draw_keyframes.push_back(tmp_buffer[i]);
			int id = tmp_buffer[i]->m_id;
			m_id_to_keyframe[id] = tmp_buffer[i];
		}
		m_change_mutex.unlock();
		tmp_buffer.swap(std::vector<c_keyframe_draw*>());
		tmp_poses.swap(std::vector<lsd_slam::c_FramePose*>());


		/*
		int nxt = m_change_flag ^ 1;

		m_poses[nxt].swap(std::vector<Sophus::Sim3>());
		int len = m_depths[nxt].size();
		for (int i = 0; i < len; i++)
		{
			lsd_slam::c_FrameMemory::get_instance().reclaim_buffer(m_depths[nxt][i]);
		}
		m_depths[nxt].swap(std::vector<float*>());

		m_graph_pt->m_pose_consistency_mutex.lock_shared();
		m_graph_pt->m_keyframes_all_mutex.lock_shared();
		for (auto KF : m_graph_pt->m_keyframes_all)
		{
			Sophus::Sim3 pose = KF->get_scaled_cam_to_world();
			float* tmp = (float*)lsd_slam::c_FrameMemory::get_instance().dispatch_buffer(sizeof(float) * m_width * m_height);
			memcpy(tmp, KF->get_idepth(), sizeof(float) * m_width * m_height);
			m_poses[nxt].push_back(pose);
			m_depths[nxt].push_back(tmp);
		}
		m_graph_pt->m_keyframes_all_mutex.unlock_shared();
		m_graph_pt->m_pose_consistency_mutex.unlock_shared();

		if (m_points[nxt] != nullptr)
			free(m_points[nxt]);

		size_t points_number = m_width * m_height * m_depths[nxt].size() * 3;
		m_points[nxt] = (float*)malloc(sizeof(float) * points_number);

		lsd_slam::c_index_thread_reduce* map_reducer = new lsd_slam::c_index_thread_reduce(16);
		for (m_change_id = 0; m_change_id < m_depths[nxt].size(); m_change_id++)
		{
			map_reducer->reduce(std::bind(&slam_gl::c_slam_draw::do_draw_change_row, this, std::placeholders::_1, std::placeholders::_2), 0, m_height,0);
			data.camera_change_tmp = data.camera_change_tmp / (m_width * m_height);
			for(int j = 0;j < 3;j++)
				data.camera.center[j] = (data.camera.center[j] + data.camera_change_tmp[j])/2;
			data.camera_change_tmp.setZero();
		}


		m_change_id = 0;
		len = m_depths[nxt].size();
		for (int i = 0; i < len; i++)
		{
			lsd_slam::c_FrameMemory::get_instance().reclaim_buffer(m_depths[nxt][i]);
		}
		m_depths[nxt].swap(std::vector<float*>());

		m_change_mutex.lock();
		m_change_flag = nxt;
		int pre = nxt ^ 1;
		free(m_points[pre]);
		m_points[pre] = nullptr;
		m_change_mutex.unlock();
		*/


	}
	
	/*
	void c_slam_draw::do_draw_change_row(int ymin, int ymax)
	{
		unsigned int pos = m_width * m_height * m_change_id*3;
		int nxt = m_change_flag ^ 1;
		float* target = m_points[nxt];
		float* source = m_depths[nxt][m_change_id];
		Sophus::Sim3 pose = m_poses[nxt][m_change_id];
		Eigen::Matrix3f rotation = pose.rxso3().matrix().cast<float>();
		Eigen::Vector3f translation = pose.translation().cast<float>();
		Eigen::Matrix3f KInv = m_K.inverse();
		float fxi = KInv(0, 0);
		float fyi = KInv(1, 1);
		float cxi = KInv(0, 2);
		float cyi = KInv(1, 2);

		float sum_idepth = 0;
		float sum_x = 0;
		float sum_y = 0;
		for(int y = ymin;y < ymax;y++)
			for (int x = 0; x < m_width; x++)
			{
				int index = x + y * m_width;
				Eigen::Vector3f point = Eigen::Vector3f(fxi * x + cxi, fyi * y + cyi, 1.0f);
				float d = *(source + index);
				if (d == -1)
					continue;
				point = point / d;
				//Eigen::Vector3f target_point = rotation.inverse() * (point - translation);
				Eigen::Vector3f target_point = rotation * point + translation;
				sum_x += target_point[0];
				sum_y += target_point[1];
				sum_idepth += target_point[2];
				for (int j = 0; j < 3; j++)
					*(target + pos + index * 3 + j) = target_point(j);
			}
		data.camera_mutex.lock();
		data.camera_change_tmp = data.camera_change_tmp + Eigen::Vector3f(sum_x,sum_y, sum_idepth);
		data.camera_mutex.unlock();
	}
	*/

	void c_slam_draw::do_draw()
	{
		glfwInit();
		glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
		glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

		GLFWwindow* window = glfwCreateWindow(data.SCR_WIDTH, data.SCR_HEIGHT, "OpenGL", NULL, NULL);
		if (window == NULL)
		{
			std::cout << "Failed to create GLFW window" << std::endl;
			glfwTerminate();
			return;
		}
		glfwMakeContextCurrent(window);
		glfwSetCursorPosCallback(window, mouse_callback);
		glfwSetMouseButtonCallback(window, mouse_button_callback);
		glfwSetScrollCallback(window, scroll_callback);
		glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
		if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
		{
			std::cout << "Failed to initialize GLAD" << std::endl;
			return;
		}
		
		Shader shader_p("D:\\lsd_slam\\my_code_2019\\src\\gl\\VertexShader_p.glsl", "D:\\lsd_slam\\my_code_2019\\src\\gl\\FragmentShader_p.glsl");
		Shader shader_cam("D:\\lsd_slam\\my_code_2019\\src\\gl\\VertexShader.glsl", "D:\\lsd_slam\\my_code_2019\\src\\gl\\FragmentShader.glsl");



		while (!glfwWindowShouldClose(window))
		{

			float currentFrame = (float)glfwGetTime();
			data.deltaTime = currentFrame - data.lastFrame;
			data.lastFrame = currentFrame;

			glClearColor(0.2f, 0.3f, 0.3f, 1.0f);

			glClear(GL_COLOR_BUFFER_BIT);

			glViewport(0, 0, data.SCR_WIDTH, data.SCR_HEIGHT);

			m_change_mutex.lock();
			int tot = m_draw_keyframes.size();

			Eigen::Vector3f tmp_center; tmp_center.setZero();
			int tot_point = 0;
			int start = (tot > gl_threshold.cut_firstN_KF) ? gl_threshold.cut_firstN_KF : 0;
			for (int i = start; i < tot; i++)
			{
				m_draw_keyframes[i]->set_gl_buffer();

				tot_point += m_draw_keyframes[i]->m_vertex_num;
			}
			for (int i = start; i < tot; i++)
			{
				Eigen::Matrix4f pose = m_draw_keyframes[i]->m_cam_to_world.matrix().cast<float>();
				Eigen::Vector4f tmp_p;
				for (int j = 0; j < 3; j++)
					tmp_p[j] = m_draw_keyframes[i]->m_center[j];
				tmp_p[3] = 1.0f;
				tmp_p = pose * tmp_p;
				
				for(int j = 0;j < 3;j++)
					tmp_center[j] += tmp_p[j] * m_draw_keyframes[i]->m_vertex_num / tot_point;
				
			}
			for (int j = 0; j < 3; j++)
				data.camera.center[j] = tmp_center[j];

			glm::mat4 model = glm::mat4(1.0f);
			model = glm::translate(model, data.camera.center);
			glm::qua<float> q = glm::qua<float>((glm::vec3(float(data.xangle), float(data.yangle), 0.0f)));
			model = model * glm::mat4_cast(q);
			model = glm::translate(model, -data.camera.center);

			glm::mat4 projection = glm::perspective(glm::radians(data.camera.Zoom), (float)data.SCR_WIDTH / (float)data.SCR_HEIGHT, 0.1f, 100.0f);
			
			glm::mat4 view = data.camera.GetViewMatrix();
			
			for (int i = 0; i < tot; i++)
			{
				Eigen::Matrix4f tmp_pose = m_draw_keyframes[i]->m_cam_to_world.matrix().cast<float>();
				shader_p.use();
				shader_p.setMat4("cameraPose", tmp_pose);
				shader_p.setMat4("model", model);
				shader_p.setMat4("projection", projection);
				shader_p.setMat4("view", view);
				

				//glEnable(GL_PROGRAM_POINT_SIZE);

				if (i >= start)
				{

					glBindVertexArray(m_draw_keyframes[i]->m_gl_VAO_id);
					glBindBuffer(GL_ARRAY_BUFFER, m_draw_keyframes[i]->m_gl_buffer_id);

					glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
					glEnableVertexAttribArray(0);

					glDrawArrays(GL_POINTS, 0, m_draw_keyframes[i]->m_vertex_num);
				}


				
				shader_cam.use();
				shader_cam.setMat4("cameraPose", tmp_pose);
				shader_cam.setMat4("model", model);
				shader_cam.setMat4("projection", projection);
				shader_cam.setMat4("view", view);

				
				m_draw_keyframes[i]->draw_cam(m_draw_keyframes[i]->m_id == m_current_cam->m_id);

				glBindVertexArray(m_draw_keyframes[i]->m_gl_cam_VAO_id);
				glBindBuffer(GL_ARRAY_BUFFER, m_draw_keyframes[i]->m_gl_cam_vertex_id);

				glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)0);
				glEnableVertexAttribArray(0);

				glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)(3 * sizeof(float)));
				glEnableVertexAttribArray(1);

				glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)(6 * sizeof(float)));
				glEnableVertexAttribArray(2);

				glDrawElements(GL_LINES, 8*2, GL_UNSIGNED_INT, 0);

				
			}

			Eigen::Matrix4f current_pose = m_current_cam->m_cam_to_world.matrix().cast<float>();
			shader_cam.use();
			shader_cam.setMat4("cameraPose", current_pose);
			shader_cam.setMat4("model", model);
			shader_cam.setMat4("projection", projection);
			shader_cam.setMat4("view", view);
			m_current_cam->draw_cam(true);
			glBindVertexArray(m_current_cam->m_gl_cam_VAO_id);
			glBindBuffer(GL_ARRAY_BUFFER, m_current_cam->m_gl_cam_vertex_id);
			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)0);
			glEnableVertexAttribArray(0);
			glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)(3 * sizeof(float)));
			glEnableVertexAttribArray(1);
			glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)(6 * sizeof(float)));
			glEnableVertexAttribArray(2);
			glDrawElements(GL_LINES, 8 * 2, GL_UNSIGNED_INT, 0);

			m_change_mutex.unlock();

			
			/*
			m_change_mutex.lock();
			int flag = m_change_flag;
			unsigned int tot = m_width * m_height * m_poses[flag].size() * 3;
			glBindVertexArray(VAO);
			glBindBuffer(GL_ARRAY_BUFFER, VBO);
			glBufferData(GL_ARRAY_BUFFER, tot * sizeof(float), m_points[flag], GL_STATIC_DRAW);
			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
			glEnableVertexAttribArray(0);	

			//glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);

			m_change_mutex.unlock();
			

			glDrawArrays(GL_POINTS, 0, tot);
			*/

			glfwSwapBuffers(window);
			glfwPollEvents();
		}

		glfwTerminate();
		return;

	}

	void c_slam_draw::change_current_cam(Sophus::Sim3 pose, int id)
	{
		m_change_mutex.lock();
		m_current_cam->set_from_pose(pose, id);
		m_change_mutex.unlock();
	}
}