#pragma once 
#include <vector>
#include <mutex>
#include <condition_variable>
#include <deque>

#include "util/settings.h"
#include "util/Sophus_util.h"
#include "Tracking/c_Relocalizer.h"

namespace lsd_slam
{
    class c_TrackingReference;
    class c_SE3Tracker;
    class c_DepthMap;
    class c_Frame;
    class c_KeyFrameGraph;

    class c_DepthMapEstimation
    {
        public:
            c_DepthMapEstimation(int w,int h,Eigen::Matrix3f K,c_KeyFrameGraph* graph);
            ~c_DepthMapEstimation();

            //init
            void random_init(unsigned char* image,int id);
            void get_depth_init(unsigned char* image, float* depth,int id);
            
            //mapping
            bool do_mapping();

            void finish_current_KF();
            void discard_current_KF();


            void create_new_current_KF(std::shared_ptr<c_Frame> new_KF_candidate);
            void load_new_current_KF_from_exist(c_Frame* frame);

            bool update_KF();


            void take_relocalize_result();


            //set and get
            void set_current_KF(std::shared_ptr<c_Frame>& frame);

            void set_tracking_is_good(bool flag);

            void set_last_tracked_frame(c_Frame* frame);

            int m_width,m_height;
            Eigen::Matrix3f m_K;

            std::mutex m_tracking_is_good_mutex;
            bool m_tracking_is_good;

            std::mutex m_last_tracked_frame_mutex;
            std::shared_ptr<c_Frame> m_last_tracked_frame;
            bool m_should_create_new_KF;

            c_TrackingReference* m_tracking_reference;
            c_SE3Tracker* m_se3_tracker;

            c_DepthMap* m_map;
            c_TrackingReference* m_mapping_reference;

            c_Relocalizer* m_relocalizer;

            std::condition_variable m_new_mapped_frame_signal;
            std::mutex m_new_mapped_frame_mutex;


            std::condition_variable m_unmapped_tracked_frames_signal;
            std::deque<std::shared_ptr<c_Frame> > m_unmapped_tracked_frames;
            std::mutex m_unmapped_tracked_frames_mutex;

            std::shared_ptr<c_Frame> m_current_keyframe;
            std::shared_ptr<c_Frame> m_tracking_reference_shared_PT;
            std::mutex m_current_keyframe_mutex;


            c_KeyFrameGraph* m_keyframe_graph; 
    };
}
