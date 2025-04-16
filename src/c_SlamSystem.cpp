#include "c_SlamSystem.h"

#include "DataStructures/c_Frame.h"
#include "DataStructures/c_FrameMemory.h"
#include "GlobalMapping/c_MapOptimization.h"
#include "DepthEstimation/c_DepthMapEstimation.h"

#include "GlobalMapping/c_KeyFrameGraph.h"

#include "Tracking/c_SE3Tracker.h"
#include "Tracking/c_Sim3Tracker.h"
#include "Tracking/c_TrackingReference.h"

#include "util/settings.h"
#include "util/Sophus_util.h"
#include "D:\slam_all\init\my_code\test_fabmap\adalam.cpp"

using namespace lsd_slam;

c_SlamSystem::c_SlamSystem(int w,int h,Eigen::Matrix3f K)
{
    //todo
    if(w % 16 != 0 || h % 16 != 0)
    {
    }

    

    m_width = w;
    m_height = h;
    m_K = K;
    m_tracker = new c_SE3Tracker(w,h,K);

    for(int level = 4;level < PYRAMID_LEVELS;++level)
        m_tracker->m_hyperparameters.maxItsPerLvl[level] = 0;    

    m_map_optimization  = new c_MapOptimization(w,h,K);
    m_depthmap_estimation = new c_DepthMapEstimation(w,h,K,m_map_optimization->m_keyframe_graph);
    
   
    

    m_is_thread_keep_running = true;

    m_constraint_thread = std::thread(&c_SlamSystem::constraint_thread_loop,this);
    m_optimization_thread = std::thread(&c_SlamSystem::optimization_thread_loop,this);

    m_slam_draw_pt = new slam_gl::c_slam_draw(w, h, K,m_map_optimization->m_keyframe_graph);
    m_slam_gl_change_thread = std::thread(&c_SlamSystem::slam_gl_change_thread_loop, this);
    m_slam_gl_draw_thread = std::thread(&slam_gl::c_slam_draw::do_draw, m_slam_draw_pt);

    m_map_thread = std::thread(&c_SlamSystem::map_thread_loop, this);
    
}

void c_SlamSystem::random_init(unsigned char* image,int id)
{
    m_depthmap_estimation->random_init(image,id);
}

void c_SlamSystem::get_depth_init(unsigned char* image, float* depth,int id)
{
    m_depthmap_estimation->get_depth_init(image,depth,id);
}

SE3 c_SlamSystem::feature_use(cv::Mat img1,cv::Mat img2,const float* origin_idepth)
{
    Eigen::Matrix3f K = m_K;

    cv::Mat line_d1, line_d2; 
    std::vector<cv::line_descriptor::KeyLine> line_k1, line_k2;
    line_d1.zeros(0, 0, CV_32F);
    line_d2.zeros(0, 0, CV_32F);
    line_k1.clear();
    line_k2.clear();

    std::cout << "init\n";

    cv::Ptr<cv::ORB> orb = cv::ORB::create(1500);
    std::vector<cv::KeyPoint> point_k1, point_k2;
    orb->detect(img1, point_k1);
    orb->detect(img2, point_k2);
    cv::Mat point_d1, point_d2;//500*32
    orb->compute(img1, point_k1, point_d1);
    orb->compute(img2, point_k2, point_d2);

    std::cout << "orb\n";

    std::vector<int> point1_result, point2_result, line1_result, line2_result;
    adalam m_adalam = adalam(
        m_width, m_height,
        point_d1, point_d2,
        point_k1,
        point_k2,
        line_d1, line_d2,
        line_k1,
        line_k2);

    m_adalam.core(point1_result, point2_result, line1_result, line2_result);
    int n_point = point1_result.size();
    int n_line = line1_result.size();

    std::cout << "adalam\n";

    std::vector<cv::Point3d> world_points;
    std::vector<cv::Point2d> image_point;
    std::vector<cv::KeyPoint> point_t1;
    std::vector<cv::KeyPoint> point_t2;
    for (int i = 0; i < n_point; i++)
    {
        auto point1 = point_k1[point1_result[i]].pt;
        auto point2 = point_k2[point2_result[i]].pt;
        Eigen::Vector3f tmp_point(point1.x, point1.y, 1.0f);
        int index = point1.x + point1.y * m_width;
        if (origin_idepth[index] < 0 || origin_idepth[index] > 1.5)
            continue;
        tmp_point = K.inverse() * tmp_point / origin_idepth[index];
        cv::Point3d point(tmp_point[0], tmp_point[1], tmp_point[2]);
        world_points.push_back(point);
        image_point.push_back(point2);
        point_t1.push_back(point_k1[point1_result[i]]);
        point_t2.push_back(point_k2[point2_result[i]]);
    }

    std::cout << "point\n";

    if (world_points.size() < 4)
        return SE3();

    
    cv::Mat res = cv::Mat(m_width, img1.cols + img2.cols, CV_8UC3);
    cv::Mat roi1(res, cv::Rect(0, 0, img1.cols, img1.rows));
    cv::Mat roi2(res, cv::Rect(img1.cols, 0, img2.cols, img2.rows));
    cv::drawKeypoints(img1, point_t1, roi1, (0, 0, 255));
    cv::drawKeypoints(img2, point_t2, roi2, (0, 0, 255));

    for (int i = 0; i < point_t1.size(); i++)
    {
        cv::line(res, point_t1[i].pt, cv::Point(point_t2[i].pt.x + img1.cols, point_t2[i].pt.y), cv::Scalar(0, 255, 0));
    }
    //cv::imshow("test", res);
    //cv::waitKey(500);
    
    
    cv::Mat K_cv;
    Eigen::Matrix3d K_t = K.cast<double>();
    cv::eigen2cv(K, K_cv);
    cv::Mat r_cv, R_cv, t_cv;
    cv::solvePnPRansac(world_points, image_point, K_cv, cv::Mat(), r_cv, t_cv);
    std::cout << t_cv << std::endl;
    cv::Rodrigues(r_cv, R_cv);
    Eigen::Matrix3d R_eigen;
    Eigen::Vector3d t_eigen;
    cv::cv2eigen(R_cv, R_eigen);
    cv::cv2eigen(t_cv, t_eigen);
    for (int i = 0; i < world_points.size(); i++)
    {
        Eigen::Vector3d point = Eigen::Vector3d(world_points[i].x,world_points[i].y,world_points[i].z);
        point = K.cast<double>()*(R_eigen * point + t_eigen);
        point /= point[2];
        float dis = (point[0] - image_point[i].x) * (point[0] - image_point[i].x) + (point[1] - image_point[i].y) * (point[1] - image_point[i].y);
        std::cout << world_points[i].z <<' '<< dis << std::endl;
    }
    

    SE3 pose(R_eigen, t_eigen);
    return pose.inverse();
}

void c_SlamSystem::trackFrame(cv::Mat img,unsigned int frame_id)
{
    unsigned char* image = img.data;
    std::shared_ptr<c_Frame> tracking_new_frame(new c_Frame(frame_id,m_width,m_height,m_K,image));
   

    if(!m_depthmap_estimation->m_tracking_is_good)
    {
        m_depthmap_estimation->m_relocalizer->update_current_frame(tracking_new_frame);

        m_depthmap_estimation->m_unmapped_tracked_frames_mutex.lock();
        m_depthmap_estimation->m_unmapped_tracked_frames_signal.notify_one();
        m_depthmap_estimation->m_unmapped_tracked_frames_mutex.unlock();

        return;
    }

    m_depthmap_estimation->m_current_keyframe_mutex.lock();
    bool pre_should_create_new_KF = m_depthmap_estimation->m_should_create_new_KF;
    if(
        m_depthmap_estimation->m_tracking_reference->m_keyframe != m_depthmap_estimation->m_current_keyframe.get() 
        || m_depthmap_estimation->m_current_keyframe->m_has_depth_been_updated)
    {
        m_depthmap_estimation->m_tracking_reference->invalidate();

        m_depthmap_estimation->m_tracking_reference->import_frame(m_depthmap_estimation->m_current_keyframe.get());
        m_depthmap_estimation->m_current_keyframe->m_has_depth_been_updated = false;
        m_depthmap_estimation->m_tracking_reference_shared_PT = m_depthmap_estimation->m_current_keyframe;
    }

    c_FramePose* tracking_ref_pose = m_depthmap_estimation->m_tracking_reference->m_keyframe->m_pose;
    m_depthmap_estimation->m_current_keyframe_mutex.unlock();

    m_map_optimization->m_keyframe_graph->m_pose_consistency_mutex.lock_shared();
    lsd_slam::c_FramePose* pre_pose = m_map_optimization->m_keyframe_graph->m_all_frame_poses.back();
    SE3 frame_to_ref_init = se3FromSim3(
        (tracking_ref_pose->get_cam_to_world()).inverse() * (pre_pose->get_cam_to_world())
    );
    m_map_optimization->m_keyframe_graph->m_pose_consistency_mutex.unlock_shared();



    SE3 new_ref_to_frame = m_tracker->trackFrame(
        m_depthmap_estimation->m_tracking_reference,
        tracking_new_frame.get(),
        frame_to_ref_init
    );

    //if ()
    if (
        manualTrackingLossIndicated
        || m_tracker->m_diverged
        || (
            m_map_optimization->m_keyframe_graph->m_keyframes_all.size() > INITIALIZATION_PHASE_COUNT
            && !m_tracker->m_tracking_was_good)
        )
    {
        int id = m_depthmap_estimation->m_tracking_reference->m_frame_ID;
        std::string filename = m_image_files[id];
        cv::Mat origin = cv::imread(filename.c_str());
        cv::Mat tmp;
        cv::cvtColor(origin, tmp, cv::COLOR_BGR2GRAY);

        if (origin.rows > 1000)
        {
            origin = tmp;
            cv::resize(origin, tmp, cv::Size(tmp.cols / 2, tmp.rows / 2));
        }
        origin = tmp;
        img = cv::imread(m_image_files[frame_id].c_str());

        const float* origin_idepth = m_depthmap_estimation->m_tracking_reference->m_keyframe->get_idepth();
        new_ref_to_frame = feature_use(origin, img,  origin_idepth);

        SE3 frame_to_ref = new_ref_to_frame.inverse();

        if (frame_to_ref.translation()[0] == 0 && frame_to_ref.translation()[1] == 0)
            return;

        c_Frame* frame = tracking_new_frame.get();
        frame->m_pose->m_this_to_parent_raw = sim3FromSE3(frame_to_ref, 1);
        m_tracker->m_tracking_was_good = true;
        m_tracker->m_diverged = false;
        manualTrackingLossIndicated = false;
        //frame_to_ref_init = feature_use(origin, img, origin_idepth).inverse();
    }
	
    m_slam_draw_pt->change_current_cam(tracking_new_frame->m_pose->get_cam_to_world(), frame_id);
    //debug
    /*
    Eigen::Vector3d pos = new_ref_to_frame.translation();
    Eigen::Matrix3d rot = new_ref_to_frame.rotationMatrix();
    for(int j = 0;j < 3;j++) std::cout<<pos[j]<<' ';for(int j = 0;j < 3;j++) for(int k = 0;k < 3;k++) std::cout<<rot(j,k)<<' ';std::cout<<std::endl;
    */

    /*
    if (m_map_optimization->m_keyframe_graph->m_keyframes_all.size() >= 1)
    {
        std::cout << "lock number:"<<m_map_optimization->m_keyframe_graph->m_keyframes_all[0]->get_read_count()<<std::endl;
    }
    */
/*
    FILE* fp = fopen("/home/g107904/my_code/data/all/my","a+");
	fprintf(fp,"%d ",frame_id);
    for(int j = 0;j < 3;j++)
	fprintf(fp,"%f ",pos[j]);
    for(int j = 0;j < 3;j++)
	for(int k = 0; k < 3;k++)
		fprintf(fp,"%f ",rot(j,k));
    fprintf(fp,"\n");
    fclose(fp);

    fp = fopen("/home/g107904/my_code/data/visualize/pose.txt","a+");
    fprintf(fp,"%d %d\n",m_depthmap_estimation->m_tracking_reference->m_keyframe->get_id(),tracking_new_frame.get()->get_id());
    Sophus::Sim3f pose = m_depthmap_estimation->m_tracking_reference->m_keyframe->get_scaled_cam_to_world().cast<float>();
    Eigen::Vector3f trans = pose.translation();
    Eigen::Matrix3f rotation = pose.rotationMatrix();
    for(int j = 0;j < 3;j++)	fprintf(fp,"%f ",trans[j]);
    for(int j = 0;j < 3;j++) for(int k = 0;k < 3;k++) fprintf(fp,"%f ",rotation(j,k)); fprintf(fp,"\n");
    pose = tracking_new_frame.get()->get_scaled_cam_to_world().cast<float>();
    trans = pose.translation();
    rotation = pose.rotationMatrix();
    for(int j = 0;j < 3;j++) fprintf(fp,"%f ",trans[j]);
    for(int j = 0;j < 3;j++) for(int k = 0;k < 3;k++) fprintf(fp,"%f ",rotation(j,k)); fprintf(fp,"\n");
    fclose(fp);
*/
    /*
    if(
        manualTrackingLossIndicated 
        || m_tracker->m_diverged 
        || (
            m_map_optimization->m_keyframe_graph->m_keyframes_all.size() > INITIALIZATION_PHASE_COUNT 
            && !m_tracker->m_tracking_was_good)
        )
    {
        m_depthmap_estimation->m_tracking_reference->invalidate();

        m_depthmap_estimation->m_tracking_is_good = false;

        m_depthmap_estimation->m_unmapped_tracked_frames_mutex.lock();
        m_depthmap_estimation->m_unmapped_tracked_frames_signal.notify_one();
        m_depthmap_estimation->m_unmapped_tracked_frames_mutex.unlock();

        manualTrackingLossIndicated = false;
        return;
    }
    */
    m_map_optimization->m_keyframe_graph->add_frame(tracking_new_frame.get());

    m_depthmap_estimation->m_last_tracked_frame = tracking_new_frame;
    if(
        !pre_should_create_new_KF
        && m_depthmap_estimation->m_current_keyframe->m_num_mapped_on_this_total > MIN_NUM_MAPPED
    )
    {
        Sophus::Vector3d dist = new_ref_to_frame.translation() * m_depthmap_estimation->m_current_keyframe->m_mean_idepth;
        float min_val = fmin(1.0f,
            0.2f+m_map_optimization->m_keyframe_graph->m_keyframes_all.size() * 0.8f / INITIALIZATION_PHASE_COUNT);
        
        if(m_map_optimization->m_keyframe_graph->m_keyframes_all.size() < INITIALIZATION_PHASE_COUNT)
        {
            min_val *= 0.7;
        }

        float closeness_score = m_map_optimization->m_keyframe_graph->get_reference_frame_score(dist.dot(dist),m_tracker->m_point_usage);

        if(closeness_score > min_val)
        {
            if(!(frame_id > 2250 && frame_id < 2350))
            m_depthmap_estimation->m_should_create_new_KF = true;
        } 
    }

    /*
    if (m_depthmap_estimation->m_current_keyframe->m_num_mapped_on_this_total > 20)
    {
        if ((frame_id > 2200 &&  frame_id < 2500) || (frame_id > 3500 && frame_id < 3700) || (frame_id > 5800 && frame_id < 6000))
        {
            m_depthmap_estimation->m_should_create_new_KF = true;
        }
    }
    
    if (m_depthmap_estimation->m_current_keyframe->m_num_mapped_on_this_total > 200)
        m_depthmap_estimation->m_should_create_new_KF = true;
        */
    

    m_depthmap_estimation->m_unmapped_tracked_frames_mutex.lock();
    if(
        m_depthmap_estimation->m_unmapped_tracked_frames.size() < 50
        || (
            m_depthmap_estimation->m_unmapped_tracked_frames.size() < 100
            && tracking_new_frame->get_tracking_parent()->m_num_mapped_on_this_total < 10
            )
        )
    {
        m_depthmap_estimation->m_unmapped_tracked_frames.push_back(tracking_new_frame);
    }
        m_depthmap_estimation->m_unmapped_tracked_frames_signal.notify_one();
    m_depthmap_estimation->m_unmapped_tracked_frames_mutex.unlock();

    //m_depthmap_estimation->do_mapping();
}

void c_SlamSystem::constraint_thread_loop()
{
    std::unique_lock<std::mutex> lock(m_map_optimization->m_keyframe_graph->m_new_keyframes_mutex);
    while(m_is_thread_keep_running)
    {
        m_map_optimization->do_constraint_search(lock);
    }
}

void c_SlamSystem::optimization_thread_loop()
{
    while(m_is_thread_keep_running)
    {
        m_map_optimization->do_optimization();
    }
}

void c_SlamSystem::slam_gl_change_thread_loop()
{
    while (m_is_thread_keep_running)
    {
        m_slam_draw_pt->do_draw_change();
    }
}

void c_SlamSystem::map_thread_loop()
{
    while (m_is_thread_keep_running)
    {
        if (!m_depthmap_estimation->do_mapping())
        {
            std::unique_lock<std::mutex> lock(m_depthmap_estimation->m_unmapped_tracked_frames_mutex);
            m_depthmap_estimation->m_unmapped_tracked_frames_signal.wait_for(lock, std::chrono::milliseconds(200));
            lock.unlock();
        }
        m_depthmap_estimation->m_new_mapped_frame_mutex.lock();
        m_depthmap_estimation->m_new_mapped_frame_signal.notify_all();
        m_depthmap_estimation->m_new_mapped_frame_mutex.unlock();
    }
}

void c_SlamSystem::finalize()
{
    std::cout << "check 1\n";

    m_map_optimization->m_last_num_constraints_added_on_full_retrack = 1;
    while(m_map_optimization->m_last_num_constraints_added_on_full_retrack != 0)
    {
        m_map_optimization->m_do_full_reconstraint_track = true;
        std::this_thread::sleep_for(std::chrono::microseconds(200000));
    }

    std::cout << "check 2\n";

    m_map_optimization->m_do_final_optimization = true;
    m_map_optimization->m_new_constraint_mutex.lock();
    m_map_optimization->m_new_constraint_added = true;
    m_map_optimization->m_new_constraint_created_signal.notify_all();
    m_map_optimization->m_new_constraint_mutex.unlock();

    while(m_map_optimization->m_do_final_optimization)
    {
        std::this_thread::sleep_for(std::chrono::microseconds(200000));
    }

    //m_depthmap_estimation->m_unmapped_tracked_frames_mutex.lock();
    //m_depthmap_estimation->do_mapping();
    //m_depthmap_estimation->m_unmapped_tracked_frames_mutex.unlock();
    m_depthmap_estimation->m_unmapped_tracked_frames_mutex.lock();
    m_depthmap_estimation->m_unmapped_tracked_frames_signal.notify_one();
    m_depthmap_estimation->m_unmapped_tracked_frames_mutex.unlock();

    std::cout << "check 3\n";

    while(m_map_optimization->m_do_final_optimization)
    {
        std::this_thread::sleep_for(std::chrono::microseconds(200000));
    }

    std::unique_lock<std::mutex> lock(m_depthmap_estimation->m_new_mapped_frame_mutex);
    m_depthmap_estimation->m_new_mapped_frame_signal.wait(lock);
    m_depthmap_estimation->m_new_mapped_frame_signal.wait(lock);

    std::cout << "check 4\n";

    std::this_thread::sleep_for(std::chrono::microseconds(200000));
    m_map_optimization->m_keyframe_graph->print_all_pose();
}

c_SlamSystem::~c_SlamSystem()
{
    m_is_thread_keep_running = false;

    m_depthmap_estimation->m_new_mapped_frame_signal.notify_all();
    m_depthmap_estimation->m_unmapped_tracked_frames_signal.notify_all();
    m_map_optimization->m_keyframe_graph->m_new_keyframes_created_signal.notify_all();
    m_map_optimization->m_new_constraint_created_signal.notify_all();

    std::cout << "check 5\n";

    m_map_thread.join();

    std::cout << "check map\n";

    m_slam_gl_change_thread.join();

    std::cout << "check 6\n";

    m_slam_gl_draw_thread.join();

    std::cout << "check 7\n";

    m_constraint_thread.join();

    std::cout << "check 8\n";

    m_optimization_thread.join();

    std::cout<<"check 9\n";


    delete m_tracker;

    std::cout << "tracker\n";

    m_depthmap_estimation->m_keyframe_graph = 0;

    //m_map_optimization->m_keyframe_graph->print_all_pose();

    

    std::cout << "map op\n";
    
    delete m_depthmap_estimation;

    delete m_map_optimization;

    std::cout << "keyframegraph\n";

    c_FrameMemory::get_instance().release_buffers();
}

