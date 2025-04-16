#pragma once
#define USE_FABMAP
#ifdef USE_FABMAP
#include <opencv2/opencv.hpp>

	namespace of2 {
		class FabMap;
	}

	namespace cv {
		class BOWImgDescriptorExtractor;
	}

	namespace lsd_slam
	{

		class c_FabMap
		{
		public:
			c_FabMap();
			~c_FabMap();

			void compareAndAdd(cv::Mat keyframe, int* out_new_id, int* out_loop_id);

			bool is_valid() const;

		private:
			int m_next_image_id;

			cv::Ptr<cv::FeatureDetector> m_detector;
			cv::Ptr<cv::BOWImgDescriptorExtractor> m_bide;
			cv::Ptr<of2::FabMap> m_fabmap;

			bool m_print_confusion_matrix;
			cv::Mat m_confusionMat;

			bool m_valid;
		};
	}
#endif