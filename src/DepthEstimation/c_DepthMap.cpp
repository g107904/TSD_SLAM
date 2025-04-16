#include "DepthEstimation/c_DepthMap.h"
#include "util/settings.h"
#include "DepthEstimation/c_DepthMapPixelHypothesis.h"

#include "DataStructures/c_Frame.h"

#include "util/interpolation.h"

#include "GlobalMapping/c_KeyFrameGraph.h"

#include <stdio.h>
#include <iostream>



namespace lsd_slam
{
    c_DepthMap::c_DepthMap(int w, int h, const Eigen::Matrix3f& K)
    {
        m_width = w;
        m_height = h;
        this->m_K = K;
        m_fx = this->m_K(0,0);
	    m_fy = this->m_K(1,1);
	    m_cx = this->m_K(0,2);
	    m_cy = this->m_K(1,2);

	    this->m_KInv = m_K.inverse();
	    m_fxi = this->m_KInv(0,0);
	    m_fyi = this->m_KInv(1,1);
	    m_cxi = this->m_KInv(0,2);
	    m_cyi = this->m_KInv(1,2);
        m_activeKeyFrame = 0;
	    m_activeKeyFrameIsReactivated = false;
	    m_otherDepthMap = new c_DepthMapPixelHypothesis[m_width*m_height];
	    m_currentDepthMap = new c_DepthMapPixelHypothesis[m_width*m_height];

	    m_validityIntegralBuffer = (int*)Eigen::internal::aligned_malloc(m_width*m_height*sizeof(int));
        reset_DepthMapPixelHypothesis();

    }

    c_DepthMap::~c_DepthMap()
    {
        
        if(m_activeKeyFrame != 0)
            m_activeKeyFrame_lock->unlock_shared();
        

        delete[] m_otherDepthMap;
        delete[] m_currentDepthMap;

        Eigen::internal::aligned_free((void*)m_validityIntegralBuffer);
    }

    void c_DepthMap::invalidate()
    {
        if(m_activeKeyFrame == 0)
            return;
        m_activeKeyFrame = 0;
        m_activeKeyFrame_lock->unlock_shared();
    }

    void c_DepthMap::reset_DepthMapPixelHypothesis()
    {
        for(c_DepthMapPixelHypothesis* pt = m_otherDepthMap+m_width*m_height-1; pt >= m_otherDepthMap; pt--)
		    pt->m_isValid = false;
	    for(c_DepthMapPixelHypothesis* pt = m_currentDepthMap+m_width*m_height-1; pt >= m_currentDepthMap; pt--)
		    pt->m_isValid = false;
    }

    
    void c_DepthMap::initializeRandomly(c_Frame* frame)
    {
        m_activeKeyFrame_lock = frame->get_active_lock();
        m_activeKeyFrame = frame;
        m_activeKeyFrameImageData = m_activeKeyFrame->get_image(0);
        m_activeKeyFrameIsReactivated = false;

        const float* maxGradients = frame->get_max_gradients();

	//myDepthMap
/*
	FILE* init_out = fopen("/home/g107904/my_lsd/lsd_slam-master/lsd_slam_core/data/map/init.txt","r+");

	float* tmp_idepth = new float[640*480];
	
	int tmp_idx;
	float tmp_i;
	
	while(fscanf(init_out,"%d %f",&tmp_idx,&tmp_i) == 2)
	{
		tmp_idepth[tmp_idx] = tmp_i;
	}
	fclose(init_out);
*/
        for(int y = 1;y < m_height - 1;y++)
            for(int x = 1;x < m_width - 1;x++)
            {
                if(maxGradients[x+y*m_width] > MIN_ABS_GRAD_CREATE)
                {
                    float idepth = 0.5f + 1.0f * ((rand() % 100001) / 100000.0f); //0.5f,1.0f

		    //float idepth = tmp_idepth[x+y*m_width];
                    m_currentDepthMap[x+y*m_width] = c_DepthMapPixelHypothesis(
						idepth,
						idepth,
						VAR_RANDOM_INIT_INITIAL,
						VAR_RANDOM_INIT_INITIAL,
						20);
                }
                else
                {
                    m_currentDepthMap[x+y*m_width].m_isValid = false;
				    m_currentDepthMap[x+y*m_width].m_blacklisted = 0;
                }
            }
        m_activeKeyFrame->set_idepth_and_var(m_currentDepthMap);
        //m_activeKeyFrame_lock->unlock_shared();
    }
    

    void c_DepthMap::setFromExistingKF(c_Frame* frame)
    {
	m_activeKeyFrame_lock = frame->get_active_lock();
        m_activeKeyFrame = frame;

	const float* idepth_it = m_activeKeyFrame->get_idepth_reactive();
	const float* idepth_var_it = m_activeKeyFrame->get_idepth_var_reactive();
	const unsigned char* validity_it = m_activeKeyFrame->get_validity_reactive();

	c_DepthMapPixelHypothesis* data_it = m_currentDepthMap;
	m_activeKeyFrame->m_num_mapped_on_this = 0;
	m_activeKeyFrame->m_num_frames_tracked_on_this = 0;
	m_activeKeyFrameIsReactivated = true;
        m_activeKeyFrameImageData = m_activeKeyFrame->get_image(0);
	
        for(int y = 0;y < m_height;y++)
            for(int x = 0;x < m_width;x++)
	    {
		if(*idepth_var_it > 0)
		{
		    *data_it = c_DepthMapPixelHypothesis(*idepth_it,*idepth_var_it,*validity_it);
		}
		else
		{
		    int index = x + y * m_width;
		    m_currentDepthMap[index].m_isValid = false;
		    m_currentDepthMap[index].m_blacklisted = (*idepth_var_it == -2) ? MIN_BLACKLIST - 1 : 0;
		}
		idepth_it++;
		idepth_var_it++;
		validity_it++;
		data_it++;
	    }
	regularize(false,VAL_SUM_MIN_FOR_KEEP);
        
        //m_activeKeyFrame_lock->unlock_shared();

	
    }

    void c_DepthMap::initializeFromGroundTruth(c_Frame* frame,float* init_depth)
    {
	

        m_activeKeyFrame_lock = frame->get_active_lock();
        m_activeKeyFrame = frame;
        m_activeKeyFrameImageData = m_activeKeyFrame->get_image(0);
        m_activeKeyFrameIsReactivated = false;

        const float* maxGradients = frame->get_max_gradients();

        const float* idepth_it = m_activeKeyFrame->get_idepth(0);

	c_DepthMapPixelHypothesis* data_it = m_currentDepthMap;
        for(int y = 0;y < m_height;y++)
            for(int x = 0;x < m_width;x++)
            {
		int index = x + y * m_width;
		float tmp_idepth = 1.0f/init_depth[index];
                if(!std::isnan(tmp_idepth) && tmp_idepth > 0)
                {
                     m_currentDepthMap[index] = c_DepthMapPixelHypothesis(tmp_idepth,tmp_idepth, VAR_RANDOM_INIT_INITIAL, VAR_RANDOM_INIT_INITIAL,20);
                }
                else
                {
                    m_currentDepthMap[index].m_isValid = false;
                    m_currentDepthMap[index].m_blacklisted = 0;
                }
                
            }
	m_activeKeyFrame->set_idepth_and_var(m_currentDepthMap);
        
        //m_activeKeyFrame_lock->unlock_shared();

	

    }

    void c_DepthMap::updateKeyframe(std::deque< std::shared_ptr<c_Frame> > referenceFrames)
    {
        m_oldest_referenceFrame = referenceFrames.front().get();
	    m_newest_referenceFrame = referenceFrames.back().get();
	    m_referenceFrameByID.clear();
	    m_referenceFrameByID_offset = m_oldest_referenceFrame->get_id();

        for(std::shared_ptr<c_Frame> frame : referenceFrames)
        {
            Sim3 reference_to_KF;
            if(frame->m_pose->m_tracking_parent->m_frame_ID == m_activeKeyFrame->get_id())
			    reference_to_KF = frame->m_pose->m_this_to_parent_raw;
		    else
			    reference_to_KF = m_activeKeyFrame->get_scaled_cam_to_world().inverse() *  frame->get_scaled_cam_to_world();

		    frame->prepare_for_stereo_with(m_activeKeyFrame, reference_to_KF, m_K, 0);

		    while((int)m_referenceFrameByID.size() + m_referenceFrameByID_offset <= frame->get_id())
			    m_referenceFrameByID.push_back(frame.get());
        }
        
        
	    caculateDepth();

	//print_pointcloud();

        fillHoles();



        regularize(false,VAL_SUM_MIN_FOR_KEEP);

	if(!m_activeKeyFrame->m_has_depth_been_updated)
	{
	    m_activeKeyFrame->set_idepth_and_var(m_currentDepthMap);
	}
	
	m_activeKeyFrame->m_num_mapped_on_this++;
	m_activeKeyFrame->m_num_mapped_on_this_total++;

	//print_point_var();

	
	    

        /*
        std::string image_kf = pre_outpath+std::to_string(m_newest_referenceFrame->get_id())+"_"+"image_kf"+txtfile;
        
        FILE* out_fp = fopen(image_kf.c_str(),"w+");
        float* image_kf_data = m_activeKeyFrame->get_image(0);
        for(int j = 0;j < m_height;j++)
            for(int i = 0;i < m_width;i++)
                fprintf(out_fp,"%f\n",*(image_kf_data+i+j*m_width));
        fclose(out_fp);
        
        std::string image_ref = pre_outpath+std::to_string(m_newest_referenceFrame->get_id())+"_"+"image_ref"+txtfile;
        out_fp = fopen(image_ref.c_str(),"w+");
        float* image_ref_data = m_oldest_referenceFrame->get_image(0);
        for(int j = 0;j < m_height;j++)
            for(int i = 0;i < m_width;i++)
                fprintf(out_fp,"%f\n",*(image_ref_data+i+j*m_width));
        fclose(out_fp);
        
        std::string image_depth = pre_outpath+std::to_string(m_newest_referenceFrame->get_id())+"_"+"image_depth"+txtfile;
        out_fp = fopen(image_depth.c_str(),"w+");
        c_DepthMapPixelHypothesis* image_depth_data = m_currentDepthMap;
        for(int j = 0;j < m_height;j++)
            for(int i = 0;i < m_width;i++)
		if((*(image_depth_data+i+j*m_width)).m_isValid)
                	fprintf(out_fp,"%f\n",(*(image_depth_data+i+j*m_width)).m_idepth_smoothed);
		else
			fprintf(out_fp,"0\n");
        fclose(out_fp);
        */

    }

    void c_DepthMap::createKeyframe(c_Frame* new_frame)
    {
        assert(isValid());
        assert(new_frame!=nullptr);
        assert(new_frame->has_tracking_parent());

        c_read_write_lock* lock2 = new_frame->get_active_lock();

	//print
/*
	std::string image_kf = pre_outpath+std::to_string(new_frame->get_id())+"_"+"image_kf"+txtfile;        
        FILE* out_fp = fopen(image_kf.c_str(),"w+");
        float* image_kf_data = m_activeKeyFrame->get_image(0);
        for(int j = 0;j < m_height;j++)
            for(int i = 0;i < m_width;i++)
                fprintf(out_fp,"%f\n",*(image_kf_data+i+j*m_width));
        fclose(out_fp);
        */


        SE3 old_to_new_se3 = se3FromSim3(new_frame->m_pose->m_this_to_parent_raw).inverse();

        propagate_depth(new_frame);

        m_activeKeyFrame_lock->unlock_shared();

        m_activeKeyFrame = new_frame;
        m_activeKeyFrame_lock = m_activeKeyFrame->get_active_lock();
        m_activeKeyFrameImageData = new_frame->get_image(0);
        m_activeKeyFrameIsReactivated = false;

	
        regularize(true,VAL_SUM_MIN_FOR_KEEP);

        fillHoles();

        regularize(false,VAL_SUM_MIN_FOR_KEEP);

	
        //make mean inverse depth be one.
        float sum_d = 0,num_d = 0;
        for(c_DepthMapPixelHypothesis* source = m_currentDepthMap;source < m_currentDepthMap+m_width*m_height;source++)
        {

            if(!source->m_isValid)
            {
                continue;
            }
            sum_d += source->m_idepth_smoothed;
            num_d++;


        }
        float factor = num_d/sum_d;
        float factor2 = factor * factor;
        /*
		if(true)
			{
				std::cout.precision(8);
				std::cout<<sum_d<<' '<<num_d<<' '<<std::endl;
			}
        */
        for(c_DepthMapPixelHypothesis* source = m_currentDepthMap;source < m_currentDepthMap+m_width*m_height;source++)
        {
            if(!source->m_isValid)
            {
                continue;
            }

            source->m_idepth *= factor;
            source->m_idepth_smoothed *= factor;
            source->m_idepth_var *= factor2;
            source->m_idepth_var_smoothed *= factor2;
        }
        
	/*
	print_pointcloud();
	print_point_var();

	//print
        std::string image_ref = pre_outpath+std::to_string(new_frame->get_id())+"_"+"image_ref"+txtfile;
        out_fp = fopen(image_ref.c_str(),"w+");
        float* image_ref_data = m_activeKeyFrame->get_image(0);
        for(int j = 0;j < m_height;j++)
            for(int i = 0;i < m_width;i++)
                fprintf(out_fp,"%f\n",*(image_ref_data+i+j*m_width));
        fclose(out_fp);
        
        std::string image_depth = pre_outpath+std::to_string(new_frame->get_id())+"_"+"image_depth"+txtfile;
        out_fp = fopen(image_depth.c_str(),"w+");
        c_DepthMapPixelHypothesis* image_depth_data = m_currentDepthMap;
        for(int j = 0;j < m_height;j++)
            for(int i = 0;i < m_width;i++)
		if((*(image_depth_data+i+j*m_width)).m_isValid)
                	fprintf(out_fp,"%f\n",(*(image_depth_data+i+j*m_width)).m_idepth_smoothed);
		else
			fprintf(out_fp,"0\n");
        fclose(out_fp);
	*/

        //update frame's
        m_activeKeyFrame->m_pose->m_this_to_parent_raw = sim3FromSE3(old_to_new_se3.inverse(),factor);
        m_activeKeyFrame->m_pose->invalidateCache();
        m_activeKeyFrame->set_idepth_and_var(m_currentDepthMap);

        lock2->unlock_shared();
        //m_activeKeyFrame_lock->unlock_shared();
    }

    void c_DepthMap::finalizeKeyFrame()
    {
        assert(isValid());

        fillHoles();

        regularize(false,VAL_SUM_MIN_FOR_KEEP);

        //update keyframe data
        m_activeKeyFrame->set_idepth_and_var(m_currentDepthMap);
        //m_activeKeyFrame->calculateMeanInformation();
        m_activeKeyFrame->set_reactive_data(m_currentDepthMap);

    }

    void c_DepthMap::propagate_depth(c_Frame* new_frame)
    {
        //reset otherDepthMap
        for(int i = 0;i < m_width*m_height;i++)
        {
            c_DepthMapPixelHypothesis* it = m_otherDepthMap+i;
            it->m_isValid = false;
            it->m_blacklisted = 0;
        }

        //pose and image data
        SE3 old_to_new_se3 = se3FromSim3(new_frame->m_pose->m_this_to_parent_raw).inverse();
        Eigen::Vector3f trans_t = old_to_new_se3.translation().cast<float>();
        Eigen::Matrix3f trans_R = old_to_new_se3.rotationMatrix().matrix().cast<float>();

        const bool* tracking_was_good = 
            new_frame->get_tracking_parent() == m_activeKeyFrame ? new_frame->get_ref_pixel_was_good_no_create() : 0;

        const float* active_kf_image_data = m_activeKeyFrame->get_image(0);
        const float* new_kf_maxgrad = new_frame->get_max_gradients(0);
        const float* new_kf_image_data = new_frame->get_image(0);

        for(int y = 0;y < m_height;y++)
            for(int x = 0;x < m_width;x++)
            {
                int idx = x + y * m_width;
                c_DepthMapPixelHypothesis* source = m_currentDepthMap + idx;
                if(!source->m_isValid)
                    continue;

                Eigen::Vector3f p_new = 
                    (trans_R * Eigen::Vector3f(x*m_fxi+m_cxi,y*m_fyi+m_cyi,1.0f))  / source->m_idepth_smoothed + trans_t;

                float new_idepth = 1.0f / p_new[2];

                float u_new = p_new[0] * new_idepth * m_fx + m_cx;
                float v_new = p_new[1] * new_idepth * m_fy + m_cy;

                //check if trans point is in image
                if(!(u_new > 2.1f && v_new > 2.1f && u_new < m_width-3.1f && v_new < m_height-3.1f))
                {
                    continue;
                }




                int new_idx = (int)(u_new+0.5f)+((int)(v_new+0.5f))*m_width;
                float cur_abs_grad = new_kf_maxgrad[new_idx];

                //check if tracking is good and grad 
                if(tracking_was_good != 0)
                {
                    if(!tracking_was_good[(x >> SE3TRACKING_MIN_LEVEL) + (m_width >> SE3TRACKING_MIN_LEVEL)*(y >> SE3TRACKING_MIN_LEVEL)] || cur_abs_grad < MIN_ABS_GRAD_DECREASE)
                    {
                        continue;
                    }
                }
                else
                {
                    //caculate residual 
                    float source_color = active_kf_image_data[idx];
                    float dest_color = get_interpolated_element<float>(new_kf_image_data,u_new,v_new,m_width);
                    float residual = dest_color - source_color;

                    //check residual and grad
                    if((residual * residual > MAX_DIFF_CONSTANT + MAX_DIFF_GRAD_MULT*cur_abs_grad*cur_abs_grad) || cur_abs_grad < MIN_ABS_GRAD_DECREASE)
                    {
                        continue;
                    }
                }

                c_DepthMapPixelHypothesis* target = m_otherDepthMap+new_idx;

                //\sigma_{d_1}^2 = {\frac{d_1}{d_0}}^4 * \sigma_{d_0}^2 + \sigma_p^2
                float factor4 = new_idepth / source->m_idepth_smoothed;
                factor4 = factor4 * factor4;
                factor4 = factor4 * factor4;

                float new_var = factor4 * source->m_idepth_var;



                //check occlusion(more than one trans to this point)
                if(target->m_isValid)
                {
                    float diff = target->m_idepth - new_idepth;
                    //if diff > \sigma ,remove the far one
                    if(diff*diff*DIFF_FAC_PROP_MERGE > new_var + target->m_idepth_var)
                    {
                        if(new_idepth < target->m_idepth)
                        {
                            continue;
                        }
                        else
                        {
                            target->m_isValid = false;
                        }
                    }
                }

                if(!target->m_isValid)
                {
                    *target = c_DepthMapPixelHypothesis(new_idepth,new_var,source->m_validity_counter);
                }
                else 
                {
                    //merge two trans points
                    float weight = new_var / (target->m_idepth_var+new_var);
                    float merged_new_idepth = weight * target->m_idepth + (1.0f - weight) * new_idepth;
                    
                    float merged_validity = source->m_validity_counter + target->m_validity_counter;

                    if(merged_validity > VALIDITY_COUNTER_MAX + (VALIDITY_COUNTER_MAX_VARIABLE))
                        merged_validity = VALIDITY_COUNTER_MAX + (VALIDITY_COUNTER_MAX_VARIABLE);

                    float merged_new_var = 1.0f /(1.0f / target->m_idepth_var+1.0f/new_var);
                    *target = c_DepthMapPixelHypothesis(merged_new_idepth,merged_new_var,merged_validity);

                }
            }

            std::swap(m_currentDepthMap,m_otherDepthMap);
    }







    void c_DepthMap::caculateDepth()
    {
        const float* keyFrameMaxGradBuf = m_activeKeyFrame->get_max_gradients(0);
        
        int num_of_success = 0;

        for(int y = 3;y < m_height-3;y++)
            for(int x = 3;x < m_width-3;x++)
            {
                int idx = x + y * m_width;
                c_DepthMapPixelHypothesis* target = m_currentDepthMap + idx;
                bool hasHypothesis = target->m_isValid;

                if(hasHypothesis && keyFrameMaxGradBuf[idx] < MIN_ABS_GRAD_DECREASE)
                {
                    target->m_isValid = false;
                    continue;
                }
                
                if(keyFrameMaxGradBuf[idx] < MIN_ABS_GRAD_CREATE || target->m_blacklisted < MIN_BLACKLIST)
                {
                    continue;
                }



                bool success;
                if(!hasHypothesis)
                    success = createDepth(x,y);
                else
                    success = updateDepth(x,y);
                if(success)
                    num_of_success++;



            }

    }


    bool c_DepthMap::createDepth(int& x,int &y)
    {
        int idx = x + y * m_width;
	    
        //std::string out_step_file = pre_outpath+std::to_string(m_newest_referenceFrame->id())+"_map/"+std::to_string(idx)+txtfile;
	    //outfp[idx] = fopen(out_step_file.c_str(),"w+");

        c_DepthMapPixelHypothesis* target =  m_currentDepthMap+idx;
        c_Frame* frame = m_activeKeyFrameIsReactivated ? m_newest_referenceFrame:m_oldest_referenceFrame;

        //check good tracking

        if(frame->get_tracking_parent() == m_activeKeyFrame)
	    {
		    bool* wasGoodDuringTracking = frame->get_ref_pixel_was_good_no_create();
		    if(wasGoodDuringTracking != 0 && !wasGoodDuringTracking[(x >> SE3TRACKING_MIN_LEVEL) + (m_width >> SE3TRACKING_MIN_LEVEL)*(y >> SE3TRACKING_MIN_LEVEL)])
		    {
			    //fclose(outfp[idx]);
			    return false;
		    }
	    }

        float epl_x,epl_y;
        bool is_good = makeAndCheckEPL(x,y,frame,epl_x,epl_y);


	
	    //fprintf(outfp[idx],"%d\n",is_good);
        if(!is_good)
        {
	        //fclose(outfp[idx]);
            return false;
        }
        float result_idepth;
        float result_idepth_var;
        float result_length_EPL;//caculate for skip number of frame
        float result_error = caculate_idepth_and_var(
            x,y,epl_x,epl_y,  //index,epl_vector of keyframe
            0.0f,1.0f,1.0f/MIN_DEPTH, //min_idepth,initial_idepth,max_idepth
            frame,frame->get_image(0), //reference_frame data
            result_idepth,result_idepth_var,result_length_EPL //caculate result
        );//return match result error




        if(result_error == -3 || result_error == -2)
        {
            target->m_blacklisted--;
        }

	    //fclose(outfp[idx]);

        //check if error is good and var is good 
        if(result_error < 0 || result_idepth_var > MAX_VAR)
        {
            return false;
        }
	
	    //std::cout<<idx<<std::endl;

        result_idepth = UNZERO(result_idepth);
        *target = c_DepthMapPixelHypothesis(result_idepth,result_idepth_var,VALIDITY_COUNTER_INITIAL_OBSERVE);
        
        return true;
    }

    bool c_DepthMap::makeAndCheckEPL(int x,int y,c_Frame* frame,float& epl_x,float& epl_y)
    {
        int idx = x + y * m_width;

        //l = u - e_k = (x,y) - K O_r = (x,y) - K inv(R_k2r) t_k2r = (x,y) - K t_r2k
        Eigen::Vector3f O_r = m_K * frame->m_this_to_other_t;
        //float l_x = O_r[2] * x - O_r[0] ;
        //float l_y = O_r[2] * y - O_r[1] ;

	    float l_x = - m_fx * frame->m_this_to_other_t[0] + frame->m_this_to_other_t[2]*(x - m_cx);
	    float l_y = - m_fy * frame->m_this_to_other_t[1] + frame->m_this_to_other_t[2]*(y - m_cy);
	


	    //fprintf(outfp[idx],"%f %f\n",O_r[0]/O_r[2],O_r[1]/O_r[2]);

        if(std::isnan(l_x+l_y))
            return false;
        
        //check length
        float l_square_length = l_x*l_x+l_y*l_y;
        if(l_square_length < MIN_EPL_LENGTH_SQUARED)
        {
            return false;
        }

        //check epl-gradient magnitude

	    float l_factor = 1.0f / sqrt(l_square_length);
        float l_x_unit = l_x * l_factor;
        float l_y_unit = l_y * l_factor;//unit vector
        float gx = m_activeKeyFrameImageData[idx+1] - m_activeKeyFrameImageData[idx-1];
        float gy = m_activeKeyFrameImageData[idx+m_width] - m_activeKeyFrameImageData[idx-m_width];

	//my
        //float epl_grad_squared = gx * l_x_unit + gy * l_y_unit;
	//epl_grad_squared = epl_grad_squared*epl_grad_squared;

	float epl_grad_squared = gx*l_x+gy*l_y;
	epl_grad_squared = epl_grad_squared * epl_grad_squared / l_square_length;

        if(epl_grad_squared < MIN_EPL_GRAD_SQUARED)
        {
            return false;
        }

        //check epl_grad angle
        float g_square_length = gx*gx+gy*gy;
	    float g_factor = 1.0f / sqrt(g_square_length);
        float gx_unit = gx * g_factor;
        float gy_unit = gy * g_factor;
	//my
        //float cos_squared = l_x_unit*gx_unit+l_y_unit*gy_unit;
	//cos_squared = cos_squared*cos_squared;
	
	float cos_squared = epl_grad_squared / (gx*gx+gy*gy);


        if(cos_squared < MIN_EPL_ANGLE_SQUARED)
        {
            return false;
        }

        epl_x = l_x_unit;
        epl_y = l_y_unit;



        return true;

    }
    
    float c_DepthMap::caculate_idepth_and_var(
        int x,int y,float epl_x,float epl_y,  //index,epl_vector of keyframe
        float min_idepth,float initial_idepth,float max_idepth, //min_idepth,initial_idepth,max_idepth
        c_Frame* frame,float* frame_image, //reference_frame data
        float& result_idepth,float& result_idepth_var,float& result_length_EPL //caculate result
       ) //return match result error
    {
         Eigen::Vector3f KinvP = Eigen::Vector3f(m_fxi*x+m_cxi,m_fyi*y+m_cyi,1.0f);//KinvP = K^-1 * p_k
         Eigen::Vector3f p_inf = frame->m_K_other_to_this_R * KinvP;//p_inf = K* R^r_k * KinvP
         Eigen::Vector3f p_real = p_inf / initial_idepth + frame->m_K_other_to_this_t;//p_real = K * R^r_k * 1/d_k * K^-1 * p_k + K*t^r_k

         float distance_of_point_k = p_real[2] * initial_idepth;// =d_k / d_r

         float first_x = x - 2*epl_x*distance_of_point_k;
         float first_y = y - 2*epl_y*distance_of_point_k;
         float last_x = x + 2*epl_x*distance_of_point_k;
         float last_y = y + 2*epl_y*distance_of_point_k;
	
	    int idx = x + y*m_width;

        

         //check if out of image bound
         if(
             first_x <= 0 || first_x >= m_width-2 
             || first_y <= 0 || first_y >= m_height-2 
             || last_x <= 0 || last_x >= m_width-2 
             || last_y <= 0 || last_y >= m_height-2
             )
        {
	        //fprintf(outfp[idx],"-1\n");
            return -1;
        }

        //check distance bound
        if(distance_of_point_k < 0.7f && distance_of_point_k > 1.4f)
        {
	        //fprintf(outfp[idx],"-1\n");
            return -1;
        }

        float val_k[5];//order by [-2,2]
        for(int i = 0,j = -2;i < 5;i++,j++)
        {
            val_k[i] = get_interpolated_element<float>(
                m_activeKeyFrameImageData,x+j*epl_x* distance_of_point_k,y+j*epl_y*distance_of_point_k,m_width);
	        //fprintf(outfp[idx],"%f %f ",x+j*epl_x* distance_of_point_k,y+j*epl_y*distance_of_point_k);
        }
	    //fprintf(outfp[idx],"\n");


        Eigen::Vector3f p_close = p_inf+frame->m_K_other_to_this_t*max_idepth;



        //if point behind the image,change
        if(p_close[2] < 0.001f)
        {
            max_idepth = (0.001f-p_inf[2])/frame->m_K_other_to_this_t[2];
            p_close = p_inf + frame->m_K_other_to_this_t * max_idepth;//p_close[2] = 0.001f
        }



        p_close = p_close / p_close[2];



        Eigen::Vector3f p_far = p_inf + frame->m_K_other_to_this_t * min_idepth;


        //if point behind the image or closer than p_close,stop
        if(p_far[2] < 0.001f || max_idepth < min_idepth)
        {
	        //fprintf(outfp[idx],"-1\n");
            return -1;
        }
	
        p_far = p_far / p_far[2];
	    //fprintf(outfp[idx],"%f %f %f %f\n",p_close[0],p_close[1],p_far[0],p_far[1]);




        if(std::isnan(p_far[0]+p_close[0]))
        {
	        //fprintf(outfp[idx],"-1\n");
            return -4;//differ in updating process than -1
        }
        
        float dx_r = p_close[0] - p_far[0];
        float dy_r = p_close[1] - p_far[1];
        float epl_length = sqrt(dx_r*dx_r+dy_r*dy_r);


        
        //check epl_length in reference_frame
        if(!epl_length > 0 || std::isinf(epl_length))
	    {
	        //fprintf(outfp[idx],"-1\n");
            return -4;
	    }

        if(epl_length > MAX_EPL_LENGTH_CROP)
        {
            p_close[0] = p_far[0]+MAX_EPL_LENGTH_CROP * dx_r/epl_length;
            p_close[1] = p_far[1]+MAX_EPL_LENGTH_CROP * dy_r/epl_length;
        }

        dx_r *= GRADIENT_SAMPLE_DIST/epl_length;//unit vector
        dy_r *= GRADIENT_SAMPLE_DIST/epl_length;



        //extend two distance in epl
        p_far[0] -= dx_r;p_far[1] -= dy_r;
        p_close[0] += dx_r;p_close[1] += dy_r;

        //check epl_length min
        if(epl_length < MIN_EPL_LENGTH_CROP)
        {
            float padding = (MIN_EPL_LENGTH_CROP - epl_length) / 2.0f;
            p_far[0] -= dx_r * padding;p_far[1] -= dy_r * padding;
            p_close[0] += dx_r * padding;p_close[1] += dy_r * padding;
        }



        //check if inf point is out of image ,skip.
        if(
            p_far[0] <= SAMPLE_POINT_TO_BORDER 
            || p_far[0] >= m_width - SAMPLE_POINT_TO_BORDER 
            || p_far[1] <= SAMPLE_POINT_TO_BORDER 
            || p_far[1] >= m_height-SAMPLE_POINT_TO_BORDER)
        {
	        //fprintf(outfp[idx],"-1\n");
            return -1;
        }

        // if p_close is outside,move inside,test length.
        if(
            p_close[0] <= SAMPLE_POINT_TO_BORDER 
            || p_close[0] >= m_width - SAMPLE_POINT_TO_BORDER 
            || p_close[1] <= SAMPLE_POINT_TO_BORDER 
            || p_close[1] >= m_height-SAMPLE_POINT_TO_BORDER)
        {
            if(p_close[0] <= SAMPLE_POINT_TO_BORDER)
            {
                //p_close[1] += (SAMPLE_POINT_TO_BORDER-p_close[0])/dx_r * dy_r;
                //p_close[0] = SAMPLE_POINT_TO_BORDER;
		float to_add = (SAMPLE_POINT_TO_BORDER - p_close[0]) / dx_r;
		p_close[1] += to_add * dy_r;
		p_close[0] += to_add * dx_r;
            }
            else if(p_close[0] >= m_width - SAMPLE_POINT_TO_BORDER)
            {
                //p_close[1] += (m_width - SAMPLE_POINT_TO_BORDER - p_close[0]) / dx_r * dy_r;
                //p_close[0] = m_width - SAMPLE_POINT_TO_BORDER;
		float to_add = (m_width - SAMPLE_POINT_TO_BORDER - p_close[0]) / dx_r;
		p_close[1] += to_add * dy_r;
		p_close[0] += to_add * dx_r;
            }
            if(p_close[1] <= SAMPLE_POINT_TO_BORDER)
            {
                //p_close[0] += (SAMPLE_POINT_TO_BORDER - p_close[1]) / dy_r * dx_r;
                //p_close[1] = SAMPLE_POINT_TO_BORDER;
		float to_add = (SAMPLE_POINT_TO_BORDER - p_close[1]) / dy_r;
		p_close[0] += to_add * dx_r;
		p_close[1] += to_add * dy_r;
            }
            else if(p_close[1] >= m_height - SAMPLE_POINT_TO_BORDER)
            {
                //p_close[0] += (m_height - SAMPLE_POINT_TO_BORDER - p_close[1]) / dy_r * dx_r;
                //p_close[1] = m_height - SAMPLE_POINT_TO_BORDER;
		float to_add = (m_height - SAMPLE_POINT_TO_BORDER - p_close[1]) / dy_r;
		p_close[0] += to_add * dx_r;
		p_close[1] += to_add * dy_r;
            }

            float new_dx = p_close[0] - p_far[0];
            float new_dy = p_close[1] - p_far[1];
            float new_epl_length = sqrt(new_dx*new_dx+new_dy*new_dy);

if(x==359 && y == 8) {std::cout.precision(8);std::cout<<p_close[0]<<' '<<p_close[1]<<' '<<new_epl_length<<std::endl;}
            if(
                p_close[0] <= SAMPLE_POINT_TO_BORDER 
                || p_close[0] >= m_width - SAMPLE_POINT_TO_BORDER 
                || p_close[1] <= SAMPLE_POINT_TO_BORDER 
                || p_close[1] >= m_height-SAMPLE_POINT_TO_BORDER 
                || new_epl_length < 8.0f)
            {
	            //fprintf(outfp[idx],"-1\n");
                return -1;
            }
            
        }

        float val_r[5];
        float px = p_far[0];
        float py = p_far[1];
        for(int i = 0,j = -2;i<5;i++,j++)
        {
            val_r[i] = get_interpolated_element<float>(frame_image,px+j*dx_r,py+j*dy_r,m_width);
	        //fprintf(outfp[idx],"%f %f ",px+j*dx_r,py+j*dy_r);

        }
	

	

	    //fprintf(outfp[idx],"\n");

        int cnt_loop = 0;
        float match_x = -1;float match_y = -1;
        float best_match_err = 1e50;
        float second_match_err = 1e50;

        float best_match_err_pre=NAN,best_match_err_post = NAN,best_match_diff_pre=NAN,best_match_diff_post=NAN;//use for caculating gradient in subpixel process

        bool best_was_last_loop = false;

        float error_last = -1;

        float error_each[2][5];
        for(int i = 0;i < 5;i++)
            error_each[0][i] = error_each[0][i] = NAN;
        
        int cnt_best_in_loop = -1,cnt_second_in_loop = -1;

        while(
            cnt_loop == 0 ||
            (((dx_r < 0) == (px > p_close[0])) && ((dy_r < 0) == (py > p_close[1])))
        )
        {
            float tmp_error = 0;
            int flag = cnt_loop % 2;
            for(int i = 4;i >= 0;i--)
            {
                error_each[flag][i] = val_r[i] - val_k[i];
                tmp_error += error_each[flag][i] * error_each[flag][i];
            }
            if(tmp_error < best_match_err)
            {
                second_match_err = best_match_err;
                cnt_second_in_loop = cnt_best_in_loop;

                best_match_err = tmp_error;
                cnt_best_in_loop = cnt_loop;

                best_match_err_pre = error_last;
                float tmp_best_match_diff_pre = 0;
                for(int j = 4;j >= 0;j--)
                    tmp_best_match_diff_pre += error_each[0][j] * error_each[1][j];
                best_match_diff_pre = tmp_best_match_diff_pre;
                best_match_err_post = -1;
                best_match_diff_post = -1;
                match_x = px;
                match_y = py;
                best_was_last_loop = true; 


            }
            else
            {
                if(best_was_last_loop)
                {
                    best_match_err_post = tmp_error;
                    float tmp_best_match_err_post = 0;
                    for(int j = 4;j >= 0;j--)
                        tmp_best_match_err_post += error_each[0][j] * error_each[1][j];
                    best_match_diff_post = tmp_best_match_err_post;
                    best_was_last_loop = false;

                }

                if(tmp_error < second_match_err)
                {
                    second_match_err = tmp_error;
                    cnt_second_in_loop = cnt_loop;
                }
            }
            
	        //fprintf(outfp[idx],"%d %f %f ",cnt_loop,match_x,match_y);

            error_last = tmp_error;

            for(int i = 0;i < 4;i++)
                val_r[i] = val_r[i+1];
            

            px += dx_r;
            py += dy_r;

            val_r[4] = get_interpolated_element<float>(frame_image,px+2*dx_r,py+2*dy_r,m_width);
		
	        //fprintf(outfp[idx],"%f %f\n",px+2*dx_r,py+2*dy_r);

            cnt_loop++;


        }



        
        //check if error is too big
        if(best_match_err > 4.0f * MAX_ERROR_STEREO)
        {
            //fprintf(outfp[idx],"-1\n");
            return -3;
        }

        //check if best is enough small than second
        if(
            abs(cnt_best_in_loop - cnt_second_in_loop) > 1.0f 
            && best_match_err * MIN_DISTANCE_ERROR_STEREO > second_match_err)
        {
            //fprintf(outfp[idx],"-1\n");
            return -2;
        }

        bool did_subpixel = false;
        if(useSubpixelStereo)//computer subpixel best match point
        {
            //computer gradient on the best point,pre point,post point
            //E(u) = \sigma e_i^2 = \sigma (val_r-val_k)^2
            //\frac{dE(u)}{du} = 2 \sigma (e_i)'e_i
            //(e_i)'(u_0) = lim_{u->u_0} \frac{e(u)-e(u_0)}{u-u_0}
            //\frac{dE(u)}{du} = 2 \sigma 1\(u-u_0) * { (e(u)-e(u_0)) * e(u)}
            //                 = 2 \sigma 1\(u-u_0) * { e(u)^2 - e(u_0)*e(u)}
            //best_match_err_  = e(u)^2    
            //best_match_diff_ = e(u_0)*e(u)
            //u - u_0 = step * {dx_r,dy_r} = f(step) step=+- 1 
            //trans \frac{dE(u)}{du} to \frac{dE(u)}{d step}
            float grad_pre       = - (best_match_err_pre - best_match_diff_pre);
            float grad_cur_left  = + (best_match_err - best_match_diff_pre);
            float grad_cur_right = - (best_match_err - best_match_diff_post);
            float grad_post      = + (best_match_err_post - best_match_diff_post);
            
            int flag = 0;//pre is 1,post is 2

            //check if anyone is oob
            if(best_match_err_pre >= 0 && best_match_err_post >= 0)
            {
                //check  gradient consistence
                if((grad_cur_left < 0) == (grad_cur_right < 0))
                {
                    if((grad_pre < 0)  == (grad_cur_left >= 0))
                    {
                        //check if post has zero-crossing
                        if((grad_post < 0)  == (grad_cur_right < 0))
                        {
                            flag = 1;
                        }
                    }

                    else if((grad_post < 0) == (grad_cur_right >= 0))
                    {
                        flag = 2;
                    }


                    
                    if(flag == 1)//computer subpixel point in [pre,cur]
                    {
			/*
                        //to computer zero point,in step-grad Coordinate System,let pre point be (-1,grad_pre),cur point be (0,grad_cur_left)
                        //than zero point is computered by solved line 
                        float line_b = grad_cur_left;
                        float line_k = (grad_cur_left-grad_pre) / (0.0f-(-1.0f));
                        float zero_step = line_b / (-line_k);
                        match_x += zero_step * dx_r;//new_point's x
                        match_y += zero_step * dy_r;//new_point's y
                        //use second order taylor
                        //E(new_point) = E(cur_point)+\frac{dE}{d step} * zero_step + \frac{\partial^2 E}{\partial step^2} * zero_step^2
                        //\frac{dE}{d step} = 2*grad_cur_left
                        //\frac{\partial^2 E}{\partial step^2} = \frac{\frac{dE}{d step}(cur) - \frac{dE}{d step}(pre)}{step} = grad_cur_left - grad_pre
                        best_match_err = best_match_err + 2*grad_cur_left * zero_step + (grad_cur_left - grad_pre)*zero_step*zero_step;
			*/

			            float d = grad_cur_left / (grad_cur_left-grad_pre);
			            match_x -= d * dx_r;
			            match_y -= d * dy_r;
			            best_match_err = best_match_err - 2 * d * grad_cur_left - (grad_pre - grad_cur_left) * d * d;
                        did_subpixel = true;



                    }
                    else if(flag == 2)//computer subpixel point in [cur,post]
                    {
			/*
                        //same as pre
                        //one is (0,grad_cur_right) ,the other is (1,grad_post)
                        float line_b = grad_cur_right;
                        float line_k = (grad_post - grad_cur_right)/(1.0f-0.0f);
                        float zero_step = line_b / (-line_k);
                        match_x += zero_step*dx_r;
                        match_y += zero_step*dy_r;

                        best_match_err = best_match_err + 2*grad_cur_right * zero_step + (grad_post - grad_cur_right) * zero_step * zero_step;
			*/

			            float d = grad_cur_right / (grad_cur_right-grad_post);
			            match_x += d * dx_r;
			            match_y += d * dy_r;
			            best_match_err = best_match_err + 2 * d * grad_cur_right + (grad_post - grad_cur_right) * d * d;
                        did_subpixel = true;


                    }
                }
            }
        }//subpixel end



	
	    //fprintf(outfp[idx],"%d %f %f\n",did_subpixel,match_x,match_y);

        //consider sample distance
        float sample_dist = GRADIENT_SAMPLE_DIST * distance_of_point_k;

        //grad_along_line = \sigma \frac{(val_k[i+1]-val_k[i])}{sample_dist}^2 = 1\{sample_dist}^2 * \sigma {val_k[i+1]-val_k[i]}^2; 
        float grad_along_line = 0;
        for(int i = 3;i >= 0;i--)
        {
            float tmp = val_k[i+1] - val_k[i];
	        grad_along_line += tmp*tmp;

        }



        grad_along_line /= sample_dist*sample_dist;

        //check if error is enough small.
        if(best_match_err > MAX_ERROR_STEREO+sqrtf(grad_along_line)*20)
        {
            return -3;
        }


        //============computer depth =============================
        //now we have two matched points in keyframe and reference frame
        //the trans formual is 
        //R * 1/d * K^-1 * p_{keyframe} + t = 1/d_r * K^-1 * p_{reference}
        //R=[r_0,r_1,r_2]^T, d_r = 1/d * K^-1 * p_{keyframe}
        //compare x,y axis' length, use the larger one
        float tmp_idepth;
        float alpha;
        if(dx_r * dx_r > dy_r * dy_r)
        {
            //p_reference in norm plain,K^-1 * p_{reference}
            float norm_x = m_fxi * match_x + m_cxi;

            //d = frac{r_2 * K^-1 * p_{keyframe} * norm_x - r_0 * k^-1 * p_{keyframe}}{t_x - t_z * norm_x}
            float denominator = - frame->m_other_to_this_t[0] + frame->m_other_to_this_t[2] * norm_x;

            tmp_idepth = (-KinvP.dot(frame->m_other_to_this_R_row2) * norm_x  + KinvP.dot(frame->m_other_to_this_R_row0)) / denominator;
            // \alpha = \frac{\partial d}{\partial \lambda} = {\partial d / \partial x}{\partial x / \partial u}{\partial u / \partial \lambda}
            //        = {\partial d / \partial x} * fx_inv * {\partial u / \partial \lambda}
            //        = {\frac{r_2 * k^-1 * p_{keyframe} * t_x - r_0 * k^-1 * p_{keyframe} * t_z}{{t_x - t_z * norm_x}^2}} * fx_inv * dx_r
            alpha =dx_r * m_fxi * (-KinvP.dot(frame->m_other_to_this_R_row2)*frame->m_other_to_this_t[0] + KinvP.dot(frame->m_other_to_this_R_row0)*frame->m_other_to_this_t[2]) / (denominator * denominator) ; 

	

        }
        else
        {
            //same as x axis
            float norm_y = m_fyi * match_y + m_cyi;
            float denominator =  -frame->m_other_to_this_t[1] + frame->m_other_to_this_t[2] * norm_y;
            tmp_idepth = (-KinvP.dot(frame->m_other_to_this_R_row2) * norm_y  + KinvP.dot(frame->m_other_to_this_R_row1)) / denominator;
            alpha = dy_r * m_fyi * (-KinvP.dot(frame->m_other_to_this_R_row2)*frame->m_other_to_this_t[1] + KinvP.dot(frame->m_other_to_this_R_row1)*frame->m_other_to_this_t[2]) / (denominator * denominator) ; 




        }





	
        //check idepth
        if(tmp_idepth < 0)
        {
            if(!allowNegativeIdepths)
                return -2;
        }
        //============================= computer idepth var ========================
        //geometric_disparity_err = \frac{\sigma_l^2}{<g,l>^2}
        //sima_l = 1/4 * (1+r)
        float sigma_l = 0.25f * (1+frame->m_initial_tracked_residual);

        Eigen::Vector4f grad_cur = get_interpolated_element<Eigen::Vector4f>(m_activeKeyFrame->get_gradients(0), x, y, m_width);
	/*
        float gx = grad_cur[0];
        float gy = grad_cur[1];
        float g_length = sqrt(gx*gx+gy*gy);
        gx = gx  / g_length;
        gy = gy  / g_length;
        float gl_square = (gx*epl_x+gy*epl_y+DIVISION_EPS/g_length);
        gl_square = gl_square*gl_square;
        float geometric_disparity_err = sigma_l*sigma_l / gl_square;
	*/
	
	    float g_factor = grad_cur[0]*epl_x+grad_cur[1]*epl_y+DIVISION_EPS;
	    float geometric_disparity_err = sigma_l * sigma_l * (grad_cur[0]*grad_cur[0] + grad_cur[1]*grad_cur[1]) / (g_factor*g_factor);



        //photometric_disparity_err = \frac{2 * \sigma_i}{g_p^2}
        float photometric_disparity_err = 4.0f * cameraPixelNoise2 / (grad_along_line + DIVISION_EPS);




        result_idepth_var = 
            alpha * alpha * ((did_subpixel ? 0.05f : 0.5f) * sample_dist * sample_dist 
                            + geometric_disparity_err + photometric_disparity_err);
        result_idepth = tmp_idepth;
        result_length_EPL = epl_length;


        return best_match_err;
    }

    bool c_DepthMap::updateDepth(int& x,int& y)
    {
        const float* keyframe_max_grad_buf = m_activeKeyFrame->get_max_gradients(0);
        int idx = x+y*m_width;

	    //std::string out_step_file = pre_outpath+std::to_string(m_newest_referenceFrame->id())+"_map/"+std::to_string(idx)+txtfile;
	    //outfp[idx] = fopen(out_step_file.c_str(),"w+");

        c_DepthMapPixelHypothesis* target = m_currentDepthMap+idx;
        c_Frame* frame;

        //skip frame to have long epl
        if(!m_activeKeyFrameIsReactivated)
        {
            //check if it is oob
            if(target->m_nextStereoFrameMinID - m_referenceFrameByID_offset >= m_referenceFrameByID.size())
            {
		        //fprintf(outfp[idx],"0\n");
		        //fclose(outfp[idx]);
                return false;
            }

            if((int)target->m_nextStereoFrameMinID < m_referenceFrameByID_offset)
            {
                frame = m_oldest_referenceFrame;
            }
            else
            {
                frame = m_referenceFrameByID[(int)target->m_nextStereoFrameMinID-m_referenceFrameByID_offset];
            }
        }
        else
        {
            frame = m_newest_referenceFrame;
        }

        //check tracking 
        if(frame->get_tracking_parent() == m_activeKeyFrame)
        {
            bool* was_good = frame->get_ref_pixel_was_good_no_create();
            if(was_good != 0 && !was_good[((x>>SE3TRACKING_MIN_LEVEL)+(m_width>>SE3TRACKING_MIN_LEVEL)*(y>>SE3TRACKING_MIN_LEVEL))])
            {
		        //fprintf(outfp[idx],"0\n");
		        //fclose(outfp[idx]);
                return false;
            }
        }

        float epl_x,epl_y;
        bool is_good = makeAndCheckEPL(x,y,frame,epl_x,epl_y);
	    //fprintf(outfp[idx],"%d\n",is_good);


	
        if(!is_good)
        {
            //fclose(outfp[idx]);
            return false;
        }
        
        //idepth in [d - 2 \sigma,d + 2\sigma]
        float sigma = sqrt(target->m_idepth_var_smoothed);
        float min_idepth = target->m_idepth_smoothed - STEREO_EPL_VAR_FAC * sigma;
        float max_idepth = target->m_idepth_smoothed + STEREO_EPL_VAR_FAC * sigma;
        


        if(min_idepth < 0)
            min_idepth = 0;
        if(max_idepth > 1 / MIN_DEPTH)
            max_idepth = 1 / MIN_DEPTH;
        
        float result_idepth,result_idepth_var,result_length_EPL;
        
        float result_error = caculate_idepth_and_var(x,y,epl_x,epl_y,min_idepth,target->m_idepth_smoothed,max_idepth,frame,frame->get_image(0),result_idepth,result_idepth_var,result_length_EPL);


        float diff_idepth = result_idepth - target->m_idepth_smoothed;
	
	    //fclose(outfp[idx]);
        //out of bound
        if(result_error == -1)
        {
            return false;
        }

        //not good for stereo
        else if(result_error == -2)
        {
            target->m_validity_counter -= VALIDITY_COUNTER_DEC;
            if(target->m_validity_counter < 0)
                target->m_validity_counter = 0;
            target->m_nextStereoFrameMinID = 0;
            target->m_idepth_var *= FAIL_VAR_INC_FAC;
            if(target->m_idepth_var > MAX_VAR)
            {
                target->m_isValid = false;
                target->m_blacklisted--;
            }
            return false;
        }

        //not found (large error)
        else if(result_error == -3)
        {
            return false;
        }

        //arithmetic error
        else if(result_error == -4)
        {
            return false;
        }

        //inconsistent
        else if(DIFF_FAC_OBSERVE * diff_idepth * diff_idepth > result_idepth_var + target->m_idepth_var_smoothed)
        {
            target->m_idepth_var *= FAIL_VAR_INC_FAC;
            if(target->m_idepth_var > MAX_VAR)
                target->m_isValid = false;
            return false;
        }

        //success
        else
        {

            // increase var by a little (prediction-uncertainty)
            float new_var = target->m_idepth_var * SUCC_VAR_INC_FAC;

            float weight = result_idepth_var / (result_idepth_var + new_var);
            float new_idepth = (1 - weight) * result_idepth + weight * target->m_idepth;

            new_var = new_var * weight;
            
            //can only be decreased
            if(new_var < target->m_idepth_var)
                target->m_idepth_var = new_var;

            target->m_idepth = new_idepth;



            //increase validity
            target->m_validity_counter += VALIDITY_COUNTER_INC;
            
            float abs_grad = keyframe_max_grad_buf[idx];
            if(target->m_validity_counter > VALIDITY_COUNTER_MAX+abs_grad*(VALIDITY_COUNTER_MAX_VARIABLE)/255.0f)
            target->m_validity_counter = VALIDITY_COUNTER_MAX+abs_grad*(VALIDITY_COUNTER_MAX_VARIABLE)/255.0f;

            //computer skip number
            if(result_length_EPL < MIN_EPL_LENGTH_CROP)
            {
                float inc = m_activeKeyFrame->m_num_frames_tracked_on_this / (float)(m_activeKeyFrame->m_num_mapped_on_this+5);
                if(inc < 3)
                    inc = 3;

                inc += (int)(result_length_EPL * 10000) % 2;
                if(result_length_EPL < 0.5 * MIN_EPL_LENGTH_CROP)
                {
                    inc *= 3;
                }

                target->m_nextStereoFrameMinID = frame->get_id()+inc;
            }


	    //std::cout<<idx<<std::endl;

            return true;

        }
    }

    void c_DepthMap::fillHoles()
    {

        buildIntegral();

        memcpy(m_otherDepthMap,m_currentDepthMap,m_width * m_height * sizeof(c_DepthMapPixelHypothesis));

        int min_y = 3;
        int max_y = m_height - 2;

        const float* keyframe_maxgrad_buf = m_activeKeyFrame->get_max_gradients(0);
        for(int y = min_y;y < max_y;y++)
        {
            for(int x = 3;x < m_width;x++)
            {
                int idx = x+y*m_width;
                c_DepthMapPixelHypothesis* target = m_otherDepthMap + idx;


                //check hole
                if(target->m_isValid)
                    continue;
                //check maxgrad
                if(keyframe_maxgrad_buf[idx] < MIN_ABS_GRAD_DECREASE)
                    continue;
                
                int* cur_validity_int = m_validityIntegralBuffer + idx;
                int sum_validity_area = cur_validity_int[2+2*m_width] - cur_validity_int[2-3*m_width] - cur_validity_int[-3+2*m_width] + cur_validity_int[-3 - 3*m_width];
                
                //if it isn't blacklist
                if((target->m_blacklisted >= MIN_BLACKLIST && sum_validity_area > VAL_SUM_MIN_FOR_CREATE) || sum_validity_area > VAL_SUM_MIN_FOR_UNBLACKLIST)
                {
                    float tmp_idepth = 0,tmp_idepth_var = 0;
                    int cnt = 0;
                    for(int j = y-2,tj = 0;j < m_height && tj < 5;j++,tj++)
                        for(int i = x-2,ti = 0;i < m_width && ti < 5;i++,ti++)
                        {
                            int tmp_idx = i+j*m_width;
                            c_DepthMapPixelHypothesis* source = m_otherDepthMap+tmp_idx;
                            if(!source->m_isValid)
                                continue;
                            tmp_idepth += source->m_idepth / source->m_idepth_var;
                            tmp_idepth_var += 1.0f / source->m_idepth_var;
                            cnt++;


                        }


                    tmp_idepth = tmp_idepth / tmp_idepth_var;
                    tmp_idepth = UNZERO(tmp_idepth);



                    m_currentDepthMap[idx] = c_DepthMapPixelHypothesis(tmp_idepth,VAR_RANDOM_INIT_INITIAL,0);
                }
            }
        }


    }

    void c_DepthMap::buildIntegral()
    {
        
        //can accelerate
        int min_y = 0,max_y = m_height;
        int* buffer_pt = m_validityIntegralBuffer + min_y * m_width;
        c_DepthMapPixelHypothesis* pixel_pt = m_currentDepthMap+min_y*m_width;
        for(int y = min_y;y < max_y;y++)
        {
            int tmp_sum = 0;
            for(int x = 0;x < m_width;x++)
            {
                if(pixel_pt->m_isValid)
                    tmp_sum += pixel_pt->m_validity_counter;
                *(buffer_pt++) = tmp_sum;
                pixel_pt++;
            }
        }

        int* cur_row_pt = m_validityIntegralBuffer;
        int* next_row_pt = m_validityIntegralBuffer+m_width;
        for(int t = m_width;t < m_height * m_width;t++)
        {
            *(next_row_pt++) += *(cur_row_pt++);
        }
    }
    void c_DepthMap::regularize(bool remove_occlusion,int validity_num)//remove_occlusion is used to computer occlusion in creating new keyframe
    {
        memcpy(m_otherDepthMap,m_currentDepthMap,m_width * m_height * sizeof(c_DepthMapPixelHypothesis));

        int min_y = 2,max_y = m_height - 2;

        int regularize_radius = 2;
        float reg_dist_var = REG_DIST_VAR;
        
        for(int y = min_y;y < max_y;y++)
        {
            for(int x = regularize_radius;x < m_width - regularize_radius;x++)
            {
                c_DepthMapPixelHypothesis* target = m_currentDepthMap + x + y*m_width;
                c_DepthMapPixelHypothesis* tmp_target = m_otherDepthMap + x + y * m_width;




                if(!tmp_target->m_isValid)
                    continue;
                
                float tmp_idepth = 0, sum_val = 0,tmp_var = 0;
                
                int num_occlude = 0,num_not_occlude = 0;
                for(int dx = -regularize_radius;dx <= regularize_radius;dx++)
                    for(int dy = -regularize_radius;dy <= regularize_radius;dy++)
                    {
                        c_DepthMapPixelHypothesis* neighbor = tmp_target + dx + dy * m_width;



                        if(!neighbor->m_isValid)
                            continue;
                        
                        float diff = neighbor->m_idepth - tmp_target->m_idepth;
                        
                        //check if target is occluded by neighbor
                        if(diff*diff*DIFF_FAC_SMOOTHING > neighbor->m_idepth_var + tmp_target->m_idepth_var)
                        {
                            if(remove_occlusion)
                            {
                                if(neighbor->m_idepth > tmp_target->m_idepth)
                                    num_occlude++;
                            }
			                continue;
                        }

                        sum_val += neighbor->m_validity_counter;

                        if(remove_occlusion)
                            num_not_occlude++;
                        
                        float dist_fac = (float)(dx*dx+dy*dy)*reg_dist_var;
                        float cur_neighbor_var = 1/(neighbor->m_idepth_var+dist_fac);

                        tmp_idepth += neighbor->m_idepth * cur_neighbor_var;
                        tmp_var += cur_neighbor_var;


		




                    }

                    /*
                    if(x == 102 && y == 190)
                    {	
                        std::cout.precision(8);
                        std::cout<<tmp_idepth<<' '<<sum_val<<std::endl;
                    }
                    */
			
                    //check if area's validity is large enough
                    if(sum_val < validity_num)
                    {
                        target->m_isValid = false;
                        target->m_blacklisted--;
                        continue;
                    } 


                    //check if it should be occluded
                    if(remove_occlusion)
                    {
                        if(num_occlude > num_not_occlude)
                        {
                            target->m_isValid = false;
                            continue;
                        }
                    }

                    tmp_idepth = tmp_idepth / tmp_var;
                    tmp_idepth = UNZERO(tmp_idepth);

                    target->m_idepth_smoothed = tmp_idepth;
                    target->m_idepth_var_smoothed = 1.0f / tmp_var;



		
            }
        }
    }

    void c_DepthMap::print_pointcloud()
    {
        //std::string filename = "/home/g107904/my_lsd/lsd_slam-master/lsd_slam_core/data/map/"+std::to_string(m_newest_referenceFrame->get_id())+"_"+"my_pointcloud.txt";
	std::string filename = "/home/g107904/my_code/data/visualize/"+std::to_string(m_newest_referenceFrame->get_id())+"_"+"my_point_var.txt";
	if(m_activeKeyFrame->get_id() != 0 && m_newest_referenceFrame->get_id() == 13)
		filename = "/home/g107904/my_code/data/visualize/"+std::to_string(m_activeKeyFrame->get_id())+"_"+"my_point_var.txt";

        FILE* fp = fopen(filename.c_str(),"w+");
        c_DepthMapPixelHypothesis* depth = m_currentDepthMap;
	Sophus::Sim3f pose = m_activeKeyFrame->get_scaled_cam_to_world().cast<float>();
	Eigen::Vector3f trans = pose.translation();
	Eigen::Matrix3f rot = pose.rotationMatrix();
        for(int i = 0;i < m_width;i++)
            for(int j = 0;j < m_height;j++)
                {
                    Eigen::Vector3f p = Eigen::Vector3f(m_fxi*i+m_cxi,m_fyi*j+m_cyi,1.0f);
                    float d = (*(depth+i+j*m_width)).m_idepth;
                    if((*(depth+i+j*m_width)).m_isValid)
                    {
                        p = p / d;
			p = rot*p+trans;
			if(fabs(p[2]) > 5) 
				continue;
                        fprintf(fp,"%f %f %f\n",p[0],p[1],p[2]);
			//fprintf(fp,"%d %d %.8f\n",i,j,d);
                    }
                    
                }
        fclose(fp);
    }
    void c_DepthMap::print_point_var()
    {
        std::string filename = "/home/g107904/my_lsd/lsd_slam-master/lsd_slam_core/data/map/"+std::to_string(m_newest_referenceFrame->get_id())+"_"+"my_point_var.txt";

		
        FILE* fp = fopen(filename.c_str(),"w+");
        c_DepthMapPixelHypothesis* depth = m_currentDepthMap;
        for(int i = 0;i < m_width;i++)
            for(int j = 0;j < m_height;j++)
                {
                    int idx = i+j*m_width;
                    float d = (*(depth+idx)).m_idepth_var_smoothed;
                    if((*(depth+idx)).m_isValid)
                    {
                        fprintf(fp,"%d %d %.8f\n",i,j,d);
                    }
                    
                    
                }
        fclose(fp);
    }
    
    /*
    void c_DepthMap::updateFromDepthMap(DepthMapPixelHypothesis* actualDepthMap)
    {
        c_DepthMapPixelHypothesis* pt = m_currentDepthMap;
        DepthMapPixelHypothesis* ac_pt = actualDepthMap;
        for(int i = 0;i < m_width;i++)
            for(int j = 0;j < m_height;j++)
            {
                pt->m_isValid =   ac_pt->isValid;
                pt->m_blacklisted = ac_pt->blacklisted;
                pt->m_nextStereoFrameMinID = ac_pt->nextStereoFrameMinID; 
                pt->m_validity_counter = ac_pt->validity_counter;
                pt->m_idepth = ac_pt->idepth;
                pt->m_idepth_var = ac_pt->idepth_var;
                pt->m_idepth_smoothed = ac_pt->idepth_smoothed;
                pt->m_idepth_var_smoothed = ac_pt->idepth_var_smoothed;
            }
    }
    */
}
