
#include "FabMap.h"
#ifdef USE_FABMAP
	#include <fstream>
	#include <opencv2/core/core.hpp>
	#include <opencv2/xfeatures2d.hpp>
	#include "openfabmap.hpp"

	namespace lsd_slam
	{

		c_FabMap::c_FabMap()
		{
			m_valid = false;
			std::string train_data_path = "D:\\slam_all\\init\\my_code\\thirdparty\\openFabMap\\trainingdata\\StLuciaShortTraindata.yml";
			std::string vocab_path = "D:\\slam_all\\init\\my_code\\thirdparty\\openFabMap\\trainingdata\\StLuciaShortVocabulary.yml";
			std::string chowliu_path = "D:\\slam_all\\init\\my_code\\thirdparty\\openFabMap\\trainingdata\\StLuciaShortTree.yml";

			cv::FileStorage fs_training;
			fs_training.open(train_data_path, cv::FileStorage::READ);
			cv::Mat train_data;
			fs_training["BOWImageDescs"] >> train_data;
			if (train_data.empty())
			{
				std::cout << "train_data error!\n" << std::endl;
				return;
			}

			fs_training.release();

			cv::FileStorage fs_vocab;
			fs_vocab.open(vocab_path, cv::FileStorage::READ);
			cv::Mat vocab;
			fs_vocab["Vocabulary"] >> vocab;

			if (vocab.empty())
			{
				std::cout << "vocab data error!\n" << std::endl;
				return;
			}
			fs_vocab.release();

			cv::FileStorage fs_tree;
			fs_tree.open(chowliu_path, cv::FileStorage::READ);
			cv::Mat tree;
			fs_tree["ChowLiuTree"] >> tree;
			if (tree.empty())
			{
				std::cout << "tree data error!\n" << std::endl;
				return;
			}


			fs_tree.release();

			int options = 0;
			options |= of2::FabMap::SAMPLED;
			options |= of2::FabMap::CHOW_LIU;

			m_fabmap = new of2::FabMap2(tree, 0.39, 0, options);
			m_fabmap->addTraining(train_data);

			m_detector = cv::xfeatures2d::StarDetector::create(32, 10, 18, 18, 20);

			cv::Ptr<cv::DescriptorExtractor> extractor = cv::xfeatures2d::SURF::create(1000, 4, 2, false, true);

			cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("FlannBased");

			m_bide = new cv::BOWImgDescriptorExtractor(extractor, matcher);

			m_bide->setVocabulary(vocab);

			m_print_confusion_matrix = false;

			m_confusionMat = cv::Mat(0, 0, CV_32F);

			m_next_image_id = 0;
			m_valid = true;
		}

		c_FabMap::~c_FabMap()
		{
		}
	
		void c_FabMap::compareAndAdd(cv::Mat keyframe, int* out_new_id, int* out_loop_id)
		{
			cv::Mat frame;

			cv::Mat keyframe_image = keyframe;

			keyframe_image.convertTo(frame, CV_8UC1);

			cv::Mat bow;
			std::vector<cv::KeyPoint> keypoints;

			m_detector->detect(frame, keypoints);

			if (keypoints.empty())
			{
				*out_new_id = -1;
				*out_loop_id = -1;
				return;
			}

		
			m_bide->compute(frame, keypoints, bow);

			std::cout <<"bow:"<<(bow.type() == CV_32F) <<' '<< bow.cols  << std::endl;

			std::vector<of2::IMatch> matches;

			if (m_next_image_id > 0)
				m_fabmap->compare(bow, matches);
			m_fabmap->add(bow);

			*out_new_id = m_next_image_id;
			++m_next_image_id;

			const float min_loop_probability = 0.8f;
			float accumulated_probability = 0;
		
			for (auto match : matches)
			{
				if (match.imgIdx < 0)
					accumulated_probability += match.match;
				else
				{
					if (match.match >= min_loop_probability)
					{
						*out_loop_id = match.imgIdx;
						return;
					}
				}
				if (accumulated_probability > 1 - min_loop_probability)
					break;
			}

			*out_loop_id = -1;
			return;
		}

		bool c_FabMap::is_valid() const { return m_valid; }

	}
#endif