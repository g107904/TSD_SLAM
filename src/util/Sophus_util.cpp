#include "sophus/se3.hpp"
#include "sophus/sim3.hpp"


template class Eigen::Quaternion<float>;
template class Eigen::Quaternion<double>;

template class Sophus::SE3Group<float,0>;
template class Sophus::SE3Group<double,0>;

template class Sophus::Sim3Group<float,0>;
template class Sophus::Sim3Group<double,0>;