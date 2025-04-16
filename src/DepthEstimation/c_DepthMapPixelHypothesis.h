#pragma once
#include "util/settings.h"
#include "util/EigenCoreInclude.h"
namespace lsd_slam
{

class c_KeyFrameGraph;

class c_DepthMapPixelHypothesis
{
public:
	/** Flag telling if there is a valid estimate at this point.
	 * All other values are only valid if this is set to true. */
	bool m_isValid;

	/** Flag that blacklists a point to never be used - set if stereo fails repeatedly on this pixel. */
	int m_blacklisted;

	/** How many frames to skip ahead in the tracked-frames-queue. */
	float m_nextStereoFrameMinID;

	/** Counter for validity, basically how many successful observations are incorporated. */
	int m_validity_counter;

	/** Actual Gaussian Distribution.*/
	float m_idepth;
	float m_idepth_var;

	/** Smoothed Gaussian Distribution.*/
	float m_idepth_smoothed;
	float m_idepth_var_smoothed;


	inline c_DepthMapPixelHypothesis() : m_isValid(false), m_blacklisted(0) {};

	inline c_DepthMapPixelHypothesis(
			const float &my_idepth,
			const float &my_idepth_smoothed,
			const float &my_idepth_var,
			const float &my_idepth_var_smoothed,
			const int &my_validity_counter) :
			m_isValid(true),
			m_blacklisted(0),
			m_nextStereoFrameMinID(0),
			m_validity_counter(my_validity_counter),
			m_idepth(my_idepth),
			m_idepth_var(my_idepth_var),
			m_idepth_smoothed(my_idepth_smoothed),
			m_idepth_var_smoothed(my_idepth_var_smoothed) {};

	inline c_DepthMapPixelHypothesis(
			const float &my_idepth,
			const float &my_idepth_var,
			const int &my_validity_counter) :
			m_isValid(true),
			m_blacklisted(0),
			m_nextStereoFrameMinID(0),
			m_validity_counter(my_validity_counter),
			m_idepth(my_idepth),
			m_idepth_var(my_idepth_var),
			m_idepth_smoothed(-1),
			m_idepth_var_smoothed(-1) {};
};
}

