#pragma once
#include "util/EigenCoreInclude.h"
#include "util/settings.h"
#include "util/Sophus_util.h"
#include "DepthEstimation/c_DepthMapPixelHypothesis.h"
#include <string>
#include <vector>
#include "util/c_read_write_lock.h"
#include <deque>
#include <memory>


namespace lsd_slam
{

class c_DepthMapPixelHypothesis;
class c_Frame;
class c_KeyFrameGraph;

class c_DepthMap
{
public:
    c_DepthMap(int w, int h, const Eigen::Matrix3f& K);
    ~c_DepthMap();
	void invalidate();
    void reset_DepthMapPixelHypothesis();
    void initializeRandomly(c_Frame* new_frame);
    void initializeFromGroundTruth(c_Frame* kf,float* init_depth = 0);
    void updateKeyframe(std::deque< std::shared_ptr<c_Frame> > referenceFrames);

    void print_pointcloud();
    void print_point_var();
    void setFromExistingKF(c_Frame* kf);
    //void updateFromDepthMap(c_DepthMapPixelHypothesis* actualDepthMap);
    bool isValid(){return m_activeKeyFrame!=0;}
    void createKeyframe(c_Frame* new_frame);
    void finalizeKeyFrame();
private:

	Eigen::Matrix3f m_K, m_KInv;
	float m_fx,m_fy,m_cx,m_cy;
	float m_fxi,m_fyi,m_cxi,m_cyi;
	int m_width, m_height;


	c_Frame* m_activeKeyFrame;
	c_read_write_lock* m_activeKeyFrame_lock;
	const float* m_activeKeyFrameImageData;
	bool m_activeKeyFrameIsReactivated;

	c_Frame* m_oldest_referenceFrame;
	c_Frame* m_newest_referenceFrame;
	std::vector<c_Frame*> m_referenceFrameByID;
	int m_referenceFrameByID_offset;

	c_DepthMapPixelHypothesis* m_otherDepthMap;
	c_DepthMapPixelHypothesis* m_currentDepthMap;
	int* m_validityIntegralBuffer;
	FILE* outfp[640*480];
	std::string pre_outpath = "/home/g107904/my_lsd/lsd_slam-master/lsd_slam_core/data/map/";
	std::string outpath = "/home/g107904/my_lsd/lsd_slam-master/lsd_slam_core/data/map/map/";
	std::string txtfile = ".txt";

	void propagate_depth(c_Frame* new_frame);
    
    void caculateDepth();
	
	bool createDepth(int& x,int &y);

	bool makeAndCheckEPL(int x,int y,c_Frame* frame,float& epl_x,float& epl_y);
	float caculate_idepth_and_var(
           int x,int y,float epl_x,float epl_y,  //index,epl_vector of keyframe
           float min_idepth,float initial_idepth,float max_idepth, //min_idepth,initial_idepth,max_idepth
           c_Frame* frame,float* frame_image, //reference_frame data
           float& result_idepth,float& result_idepth_var,float& result_length_EPL //caculate result
       ); //return match result error
	
	bool updateDepth(int& x,int& y);

	void fillHoles();

	void buildIntegral();

	void regularize(bool remove_occlusion,int validity_num);
	
	
};
}
