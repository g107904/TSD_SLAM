#include <iostream>

#ifdef __cplusplus
extern "C"
{
#endif
#include <libavformat/avformat.h>
#include <libavutil/mathematics.h>
#include <libavutil/time.h>
#include <libavutil/samplefmt.h>
#include <libavcodec/avcodec.h>
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
#ifdef __cplusplus
}
#endif

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

#include "c_SlamSystem.h"
#include "c_initSystem.cpp"
#include <mutex>
#include <thread>
#include <vector>
#include <algorithm>
#include <Eigen/Core>
#include <chrono>


time_t start = 0, end = 0;
static int interrupt_func(void* context)
{
	end = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
	if (start != 0 && end - start > 3)
		return 1;
	return 0;
}

int test_flv()
{
	AVOutputFormat* out_format = NULL;
	AVFormatContext* in_format_context = NULL, * out_format_context = NULL;
	AVPacket packet;
	const char* in_filename, * out_filename;
	int msg, i;
	int video_index = -1;
	int frame_index = 0;
	in_filename = "rtmp://127.0.0.1/live/home";
	out_filename = "D:\\lsd_slam\\my_code_2019\\build\\x64\\Debug\\test.flv";


	//avformat_network_init();


	if ((msg = avformat_open_input(&in_format_context, in_filename, 0, 0)) < 0)
	{
		printf("error open input file\n");
		goto end;
	}

	if ((msg = avformat_find_stream_info(in_format_context, 0)) < 0)
	{
		printf("failed to receive data!\n");
		goto end;
	}

	for (i = 0; i < in_format_context->nb_streams; i++)
	{
		if (in_format_context->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO)
		{
			video_index = i;
			break;
		}
	}
	av_dump_format(in_format_context, 0, in_filename, 0);

	avformat_alloc_output_context2(&out_format_context, NULL, NULL, out_filename);

	if (!out_format_context)
	{
		printf("ERROR when create output context!\n");
		msg = AVERROR_UNKNOWN;
		goto end;
	}

	out_format = out_format_context->oformat;
	for (i = 0; i < in_format_context->nb_streams; i++)
	{
		AVStream* in_stream = in_format_context->streams[i];
		AVCodec* codec = avcodec_find_decoder(in_stream->codecpar->codec_id);
		AVStream* out_stream = avformat_new_stream(out_format_context, codec);

		if (!out_stream)
		{
			printf("failed to allocate output stream!\n");
			msg = AVERROR_UNKNOWN;
			goto end;
		}

		AVCodecContext* codec_context = avcodec_alloc_context3(codec);
		msg = avcodec_parameters_to_context(codec_context, in_stream->codecpar);

		if (msg < 0)
		{
			printf("failed to copy context from input to output stream codec context!\n");
			goto end;
		}

		codec_context->codec_tag = 0;
		if (out_format_context->oformat->flags & AVFMT_GLOBALHEADER)
		{
			codec_context->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
		}
		msg = avcodec_parameters_from_context(out_stream->codecpar, codec_context);
		if (msg < 0)
		{
			av_log(NULL, AV_LOG_ERROR, "eno:[%d] error to parameters codec parameter! \n", msg);

		}
	}

	av_dump_format(out_format_context, 0, out_filename, 1);

	if (!(out_format->flags & AVFMT_NOFILE))
	{
		msg = avio_open(&out_format_context->pb, out_filename, AVIO_FLAG_WRITE);
		if (msg < 0)
		{
			printf("ERROR when open output URL '%s'!\n", out_filename);
			goto end;
		}

	}

	msg = avformat_write_header(out_format_context, NULL);
	if (msg < 0)
	{
		printf("ERROR when open output URL\n");
		goto end;
	}

	while (1)
	{
		AVStream* in_stream, * out_stream;
		msg = av_read_frame(in_format_context, &packet);
		if (msg < 0)
			break;


		in_stream = in_format_context->streams[packet.stream_index];
		out_stream = out_format_context->streams[packet.stream_index];

		packet.pts = av_rescale_q_rnd(packet.pts, in_stream->time_base, out_stream->time_base, (AVRounding)(AV_ROUND_NEAR_INF | AV_ROUND_PASS_MINMAX));
		packet.dts = av_rescale_q_rnd(packet.dts, in_stream->time_base, out_stream->time_base, (AVRounding)(AV_ROUND_NEAR_INF | AV_ROUND_PASS_MINMAX));
		packet.pos = -1;

		if (packet.stream_index == video_index)
		{
			printf("Receive %8d video frames from input URL\n", frame_index);
			frame_index++;
		}
		msg = av_interleaved_write_frame(out_format_context, &packet);
		if (msg < 0)
		{
			printf("ERROR when mux packet\n");
			break;
		}
		av_packet_unref(&packet);
	}

	av_write_trailer(out_format_context);


end:
	avformat_close_input(&in_format_context);

	if (out_format_context && !(out_format->flags && AVFMT_NOFILE))
		avio_close(out_format_context->pb);
	avformat_free_context(out_format_context);
	if (msg < 0 && msg != AVERROR_EOF)
	{
		printf("ERROR!\n");
		return -1;
	}

	return 0;
}

void YUV420P_to_RGB32(const uchar* YUV_buffer_in, const uchar* RGB_buffer_out, int width, int height)
{
	uchar* YUV_buffer = (uchar*)YUV_buffer_in;
	uchar* RGB32_buffer = (uchar*)RGB_buffer_out;

	int channels = 3;
	int width_times_height = width * height;
	int half_width = width / 2, half_height = height / 2;
	int quad_width_times_height = half_width * half_height;


	for(int y = 0;y < height;y++)
		for (int x = 0; x < width; x++)
		{
			int index = y * width + x;
			int index_Y = y * width + x;
			int half_y = y / 2;
			int half_x = x / 2;
			int index_U = width_times_height + half_y * half_width + half_x;
			int index_V = width_times_height + quad_width_times_height + half_y * half_width + half_x;

			uchar Y = YUV_buffer[index_Y];
			uchar U = YUV_buffer[index_U];
			uchar V = YUV_buffer[index_V];

			int R = Y + 1.402 * (V - 128);
			int G = Y - 0.34413 * (U - 128) - 0.71414 * (V - 128);
			int B = Y + 1.772 * (U - 128);
			if (R < 0) R = 0;
			if (G < 0) G = 0;
			if (B < 0) B = 0;
			if (R > 255) R = 255;
			if (G > 255) G = 255;
			if (B > 255) B = 255;

			RGB32_buffer[index * channels + 2] = uchar(R);
			RGB32_buffer[index * channels + 1] = uchar(G);
			RGB32_buffer[index * channels + 0] = uchar(B);
		}
}
void AVFrame2Img(AVFrame* frame, cv::Mat& img)
{
	int frame_height = frame->height;
	int frame_width = frame->width;
	int channels = 3;

	img = cv::Mat::zeros(frame_height, frame_width, CV_8UC3);

	uchar* decode_buffer = (uchar*)malloc(frame_height * frame_width * sizeof(uchar) * channels);

	int width_times_height = frame_width * frame_height;
	int half_width = frame_width / 2, half_height = frame_height / 2;
	int quad_width_times_height = half_width * half_height;

	for (int i = 0; i < frame_height; i++)
	{
		memcpy(decode_buffer + frame_width * i,
			frame->data[0] + frame->linesize[0] * i,
			frame_width);
	}

	for (int j = 0; j < half_height; j++)
	{
		memcpy(decode_buffer + width_times_height + half_width * j,
			frame->data[1] + frame->linesize[1] * j,
			half_width);
	}
	for (int k = 0; k < half_height; k++)
	{
		memcpy(decode_buffer + width_times_height + quad_width_times_height + half_width * k,
			frame->data[2] + frame->linesize[2] * k,
			half_width);
	}

	/*
	cv::Mat YUV_in = cv::Mat::zeros(frame_height * 3 / 2, frame_width, CV_8UC1);
	memcpy(YUV_in.data, decode_buffer, width_times_height);
	memcpy(YUV_in.data, decode_buffer + width_times_height, quad_width_times_height);
	memcpy(YUV_in.data, decode_buffer + width_times_height + quad_width_times_height, quad_width_times_height);
	cv::cvtColor(YUV_in, img, CV_YUV2BGR_I420);
	*/
	IplImage* image = cvCreateImage(cvSize(frame_width, frame_height), IPL_DEPTH_8U, 1);
	cvSetData(image, decode_buffer, frame_width);


	//YUV420P_to_RGB32(decode_buffer, img.data, frame_width, frame_height);
	//cv::imshow("test", img);

	cvShowImage("image", image);
	
	cv::waitKey(1);

	free(decode_buffer);
	img.release();
}

int test_rtmp()
{
	AVFilterContext* buffer_src_context;
	AVFilterGraph* filter_graph;

	AVFormatContext* in_format_context = avformat_alloc_context();
	in_format_context->interrupt_callback.callback = interrupt_func;
	in_format_context->interrupt_callback.opaque = in_format_context;

	AVPacket packet;
	AVFrame* frame = NULL;
	int msg, i;
	int video_index = -1;

	AVCodecContext* codec_context;
	AVCodec* codec;

	const AVBitStreamFilter* buffer_src = NULL;
	AVBSFContext* bsf_context;
	AVCodecParameters* codec_parameter = NULL;

	const char* in_filename = "rtmp://127.0.0.1/live/home";
	const char* out_filename = "D:\\lsd_slam\\my_code_2019\\build\\Debug\\test.h264";

	av_register_all();

	avformat_network_init();

	if ((msg = avformat_open_input(&in_format_context, in_filename, 0, 0)) < 0)
	{
		printf("Error when open input file!\n");
		return -1;
	}

	if ((msg = avformat_find_stream_info(in_format_context, 0)) < 0)
	{
		printf("Error when retrive input stream info!\n");
		return -1;
	}

	video_index = -1;
	for (i = 0; i < in_format_context->nb_streams; i++)
	{
		if (in_format_context->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO)
		{
			video_index = i;
			codec_parameter = in_format_context->streams[i]->codecpar;
		}
	}



	codec = avcodec_find_decoder(AV_CODEC_ID_H264);
	if (codec == NULL)
	{
		printf("Error when find codec!\n");
		return -1;
	}

	codec_context = avcodec_alloc_context3(codec);
	if (!codec_context)
	{
		fprintf(stderr, "Error when allocate video codec context!\n");
		exit(1);
	}

	if (avcodec_open2(codec_context, codec, NULL) < 0)
	{
		printf("Error when open codec!\n");
		return -1;
	}

	frame = av_frame_alloc();
	if (!frame)
	{
		printf("Error when allocate video frame!\n");
		exit(1);
	}

	FILE* fp_video = fopen(out_filename, "wb+");

	cv::Mat image_test;

	buffer_src = av_bsf_get_by_name("h264_mp4toannexb");
	msg = av_bsf_alloc(buffer_src, &bsf_context);
	if (msg < 0)
		return -1;
	avcodec_parameters_copy(bsf_context->par_in, codec_parameter);
	msg = av_bsf_init(bsf_context);

	int count = 0;

	while (true)
	{
		std::cout << "s\n";
		start  = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
		msg = av_read_frame(in_format_context, &packet);
		std::cout << "e\n";
		if (msg < 0)
			break;
		if (packet.stream_index == video_index)
		{
			msg = av_bsf_send_packet(bsf_context, &packet);
			if (msg < 0)
			{
				printf("Error when bsg_send_packet!\n");
				continue;
			}
			msg = av_bsf_receive_packet(bsf_context, &packet);
			if (msg < 0)
			{
				printf("Error when bsg_receive_packet!\n");
				continue;
			}
			std::cout << "Write video packet.size:" << packet.size << " pts:" << packet.pts << std::endl;

			if (packet.size)
			{
				msg = avcodec_send_packet(codec_context, &packet);
				if (msg < 0
					|| msg == AVERROR(EAGAIN)
					|| msg == AVERROR_EOF)
				{
					printf("avcodec_send_packet:%d\n", msg);
					continue;
				}

				msg = avcodec_receive_frame(codec_context, frame);
				if (msg == AVERROR(EAGAIN)
					|| msg == AVERROR_EOF)
				{
					printf("avcodec_receive_frame:%d\n", msg);
					continue;
				}

				AVFrame2Img(frame, image_test);
			}
		}
		av_packet_unref(&packet);
	}


	av_bsf_free(&bsf_context);
	fclose(fp_video);
	avformat_close_input(&in_format_context);
	if (msg < 0
		&& msg != AVERROR_EOF)
	{
		printf("Error!\n");
		return -1;
	}
	return 0;
}

int init_frame_num = 0;

class slam_wrapper 
{
public:
	slam_wrapper(int width,int height,Eigen::Matrix3f K,uchar* image_data)
	{
		m_width = width;
		m_height = height;
		m_K = K;
		m_system = new lsd_slam::c_SlamSystem(width, height, K);
		m_system->random_init(image_data, 0);
		m_initSystem = new c_initSystem(m_K);
	}
	void finalize()
	{
		m_system->finalize();
		delete m_system;
	}

	void use_slam(cv::Mat image_data, int id)
	{
		if (id < init_frame_num)
			m_initSystem->initial(m_K,image_data);
		else
			m_system->trackFrame(image_data, id);
	}

private:
	lsd_slam::c_SlamSystem* m_system;
	c_initSystem* m_initSystem;
	int m_width, m_height;
	Eigen::Matrix3f m_K;
};

class ffmpeg_processer
{
public:
	ffmpeg_processer()
	{
		m_slam_wrapper = nullptr;
		m_is_first_image = true;
		m_count = 0;
		m_is_running = true;
		m_avg_size = 0;
	}

	void ffmpeg_frame_to_image(AVFrame* frame, cv::Mat& img)
	{
		int frame_height = frame->height;
		int frame_width = frame->width;
		int channels = 3;

		uchar* decode_buffer = (uchar*)malloc(frame_height * frame_width * sizeof(uchar) * channels);

		int width_times_height = frame_width * frame_height;
		int half_width = frame_width / 2, half_height = frame_height / 2;
		int quad_width_times_height = half_width * half_height;

		for (int i = 0; i < frame_height; i++)
		{
			memcpy(decode_buffer + frame_width * i,
				frame->data[0] + frame->linesize[0] * i,
				frame_width);
		}

		for (int j = 0; j < half_height; j++)
		{
			memcpy(decode_buffer + width_times_height + half_width * j,
				frame->data[1] + frame->linesize[1] * j,
				half_width);
		}
		for (int k = 0; k < half_height; k++)
		{
			memcpy(decode_buffer + width_times_height + quad_width_times_height + half_width * k,
				frame->data[2] + frame->linesize[2] * k,
				half_width);
		}
		IplImage* image = cvCreateImage(cvSize(frame_width, frame_height), IPL_DEPTH_8U, 1);
		cvSetData(image, decode_buffer, frame_width);

		cv::Mat tmp = cv::cvarrToMat(image);
		if (tmp.cols > 1000)
			cv::resize(tmp, img, cv::Size(tmp.cols / 4, tmp.rows / 4));
		else if (tmp.cols > 500)
			cv::resize(tmp, img, cv::Size(tmp.cols /2, tmp.rows /2));
		//img = tmp.clone();
		tmp.release();
		
		free(decode_buffer);
	}

	int process(const char* rtmp_server,const char* video_out_filename)
	{
		m_display_thread = std::thread(&ffmpeg_processer::display_opencv_image_thread_loop,this);
		//m_opencv_thread = std::thread(&ffmpeg_processer::opencv_thread_loop, this);
		//m_opencv_out_thread = std::thread(&rtmp_processer::output_image, this);

		m_out_path = video_out_filename;

		AVFilterContext* buffer_src_context;
		AVFilterGraph* filter_graph;

		AVFormatContext* in_format_context = avformat_alloc_context();
		in_format_context->interrupt_callback.callback = interrupt_func;
		in_format_context->interrupt_callback.opaque = in_format_context;

		AVPacket packet;
		AVFrame* frame = NULL;
		int msg, i;
		int video_index = -1;

		AVCodecContext* codec_context;
		AVCodec* codec;

		const AVBitStreamFilter* buffer_src = NULL;
		AVBSFContext* bsf_context;
		AVCodecParameters* codec_parameter = NULL;

		av_register_all();

		avformat_network_init();

		msg = avformat_open_input(&in_format_context, rtmp_server, 0, 0);
		if (msg < 0)
		{
			printf("Error when open input file!\n");
			return -1;
		}


		msg = avformat_find_stream_info(in_format_context, 0);
		if (msg < 0)
		{
			printf("Error when retrive input stream info!\n");
			return -1;
		}

		video_index = -1;
		for (i = 0; i < in_format_context->nb_streams; i++)
		{
			if (in_format_context->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO)
			{
				video_index = i;
				codec_parameter = in_format_context->streams[i]->codecpar;
			}
		}

		codec = avcodec_find_decoder(AV_CODEC_ID_H264);
		if (codec == NULL)
		{
			printf("Error when find codec!\n");
			return -1;
		}

		codec_context = avcodec_alloc_context3(codec);
		if (!codec_context)
		{
			fprintf(stderr, "Error when allocate video codec context!\n");
			exit(1);
		}

		msg = avcodec_open2(codec_context, codec, NULL);
		if (msg < 0)
		{
			printf("Error when open codec!\n");
			return -1;
		}

		frame = av_frame_alloc();
		if (!frame)
		{
			printf("Error when allocate video frame!\n");
			exit(1);
		}

		FILE* fp_video = fopen(video_out_filename, "wb+");

		buffer_src = av_bsf_get_by_name("h264_mp4toannexb");
		msg = av_bsf_alloc(buffer_src, &bsf_context);
		if (msg < 0)
			return -1;
		avcodec_parameters_copy(bsf_context->par_in, codec_parameter);
		msg = av_bsf_init(bsf_context);

		while (m_is_running)
		{
			start = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
			msg = av_read_frame(in_format_context, &packet);
			if (msg < 0)
				break;
			if (packet.stream_index == video_index)
			{
				msg = av_bsf_send_packet(bsf_context, &packet);
				if (msg < 0)
				{
					printf("Error when bsg_send_packet!\n");
					continue;
				}
				msg = av_bsf_receive_packet(bsf_context, &packet);
				if (msg < 0)
				{
					printf("Error when bsg_receive_packet!\n");
					continue;
				}
				std::cout << "Write video packet.size:" << packet.size << " pts:" << packet.pts << std::endl;

				if (packet.size)
				{
					msg = avcodec_send_packet(codec_context, &packet);
					if (msg < 0
						|| msg == AVERROR(EAGAIN)
						|| msg == AVERROR_EOF)
					{
						printf("avcodec_send_packet:%d\n", msg);
						continue;
					}

					msg = avcodec_receive_frame(codec_context, frame);
					if (msg == AVERROR(EAGAIN)
						|| msg == AVERROR_EOF)
					{
						printf("avcodec_receive_frame:%d\n", msg);
						continue;
					}
					std::unique_lock<std::mutex> lock(m_mutex);

					/*
					if ((packet.data[4]&0x1F == 1) &&  frame->pkt_size < m_avg_size)
						continue;
					*/

					ffmpeg_frame_to_image(frame, m_opencv_image);

					if (m_is_first_image)
					{
						int width = frame->width;
						int height = frame->height;
						
						if (width > 1000)
						{
							width /= 4;
							height /= 4;
						}
						else if (width > 500)
						{
							width /= 2;
							height /= 2;
						}
						
						std::cout << width << ' ' << height << std::endl;
						Eigen::Matrix3f K;
						K.setZero();
						float f = 8.8 * 0.001;
						float cmos_w = 12.8 * 0.001;
						float cmos_h = 9.6 * 0.001;
						float fx = 0.669956145342 * width;
						float cx = width / 2;
						float fy = 1.004934218014 * height;
						float cy = height / 2;

						K(0, 0) = fx;
						K(0, 2) = cx;
						K(1, 1) = fy;
						K(1, 2) = cy;
						K(2, 2) = 1;
						m_width = width;
						m_height = height;
						m_K = K;
						m_avg_size = m_avg_size * m_count + frame->pkt_size;
						m_avg_size /= (m_count + 1);

						m_slam_wrapper = new slam_wrapper(width, height, K, m_opencv_image.data);
						m_is_first_image = false;
					}
					else
					{
						if(m_count % 2 == 0)
							m_slam_wrapper->use_slam(m_opencv_image, m_count);
					}
					m_count++;
					std::cout << m_count << std::endl;
					lock.unlock();
				}
			}
			av_packet_unref(&packet);
		}

		if (m_slam_wrapper != nullptr)
		{
			m_slam_wrapper->finalize();
			delete m_slam_wrapper;
		}

		m_is_running = false;

		av_bsf_free(&bsf_context);
		fclose(fp_video);
		avformat_close_input(&in_format_context);
		if (msg < 0
			&& msg != AVERROR_EOF)
		{
			printf("Error!\n");
			return -1;
		}
		return 0;
	}

	void opencv_thread_loop()
	{
		while (m_is_running)
		{
			std::unique_lock<std::mutex> lock(m_mutex);
			m_image_count.push(m_count);
			m_images.push(m_opencv_image);
			lock.unlock();
		}
	}
	void display_opencv_image_thread_loop()
	{
		while (m_is_running)
		{
			std::unique_lock<std::mutex> lock(m_mutex);
			if (m_opencv_image.empty())
			{
				lock.unlock();
				continue;
			}
			/*
			m_queue_mutex.lock();
			m_image_count.push(m_count);
			m_images.push(m_opencv_image);
			m_queue_mutex.unlock();
			*/
			cv::imshow("image", m_opencv_image);
			cv::waitKey(1);

			m_opencv_image.release();

			lock.unlock();
		}
	}

	void output_image()
	{
		while (m_is_running || !m_images.empty())
		{
			if (m_images.empty())
				continue;
			std::unique_lock<std::mutex> lock(m_queue_mutex);
			cv::Mat top = m_images.front();
			int c = m_image_count.front();
			m_images.pop();
			m_image_count.pop();
			int len = strlen(m_out_path);
			char* out_filename = new char[len + 20];
			memcpy(out_filename, m_out_path, len*sizeof(char));
			sprintf(out_filename + len, "\\%08d.png", c);
			cv::imwrite(out_filename, top);
			lock.unlock();
		}
	}

private:
	slam_wrapper* m_slam_wrapper;
	cv::Mat m_opencv_image;

	const char* m_out_path;
	std::queue<int> m_image_count;
	std::queue<cv::Mat> m_images;
	std::mutex m_queue_mutex;
	std::thread m_opencv_out_thread;

	std::thread m_opencv_thread;
	std::thread m_display_thread;
	bool m_is_running;
	bool m_is_first_image;
	int m_count;
	std::mutex m_mutex;

	int m_avg_size;
	int m_width, m_height;
	Eigen::Matrix3f m_K;
};

int main()
{
	ffmpeg_processer* processer = new ffmpeg_processer();
	const char* in_filename = "rtmp://127.0.0.1/live";
	const char* out_filename = "D:\\lsd_slam\\my_code_2019\\img";
	processer->process(in_filename, out_filename);

}