cmake_minimum_required(VERSION 2.8.7)
project(lsd_slam_core)
# include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)

# Set the build type.  Options are:
#  Coverage       : w/ debug symbols, w/o optimization, w/ code-coverage
#  Debug          : w/ debug symbols, w/o optimization
#  Release        : w/o debug symbols, w/ optimization
#  RelWithDebInfo : w/ debug symbols, w/ optimization
#  MinSizeRel     : w/o debug symbols, w/ optimization, stripped binaries
# set(ROS_BUILD_TYPE Release)

#set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_BUILD_TYPE "Debug")

#rosbuild_init() 

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_MODULE_PATH   ${PROJECT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

find_package(Eigen3 REQUIRED)
find_package(SuiteParse REQUIRED) # Apparently needed by g2o
#find_package(X11 REQUIRED)
#find_package(OpenCV REQUIRED)

# FabMap
# uncomment this part to enable fabmap
#add_subdirectory(${PROJECT_SOURCE_DIR}/thirdparty/openFabMap)
#include_directories(${PROJECT_SOURCE_DIR}/thirdparty/openFabMap/include)
#add_definitions("-DHAVE_FABMAP")
#set(FABMAP_LIB openFABMAP )

# Dynamic Reconfigure Services
#rosbuild_find_ros_package(dynamic_reconfigure)
#include(${dynamic_reconfigure_PACKAGE_PATH}/cmake/cfgbuild.cmake)
#gencfg()

# SSE flags
#rosbuild_check_for_sse()
add_definitions("-DUSE_ROS")
add_definitions("-DENABLE_SSE")

# Also add some useful compiler flag
set(CMAKE_CXX_FLAGS
   "${CMAKE_CXX_FLAGS} ${SSE_FLAGS} -march=native  -std=c++0x"
) 

# Set source files
set(lsd_SOURCE_FILES
  ${PROJECT_SOURCE_DIR}/src/DataStructures/c_Frame.cpp
  ${PROJECT_SOURCE_DIR}/src/DataStructures/c_FramePose.cpp
  ${PROJECT_SOURCE_DIR}/src/DataStructures/c_FrameMemory.cpp

  ${PROJECT_SOURCE_DIR}/src/DepthEstimation/c_DepthMap.cpp
  ${PROJECT_SOURCE_DIR}/src/DepthEstimation/c_DepthMapPixelHypothesis.cpp
  ${PROJECT_SOURCE_DIR}/src/DepthEstimation/c_DepthMapEstimation.cpp
  ${PROJECT_SOURCE_DIR}/src/util/Sophus_util.cpp
  ${PROJECT_SOURCE_DIR}/src/util/settings.cpp
  ${PROJECT_SOURCE_DIR}/src/util/c_read_write_lock.cpp
  ${PROJECT_SOURCE_DIR}/src/Tracking/c_Sim3Tracker.cpp 
  ${PROJECT_SOURCE_DIR}/src/Tracking/c_Relocalizer.cpp
  ${PROJECT_SOURCE_DIR}/src/Tracking/c_SE3Tracker.cpp 
  ${PROJECT_SOURCE_DIR}/src/Tracking/c_TrackingReference.cpp 
  ${PROJECT_SOURCE_DIR}/src/GlobalMapping/c_KeyFrameGraph.cpp
  ${PROJECT_SOURCE_DIR}/src/GlobalMapping/g2o_with_type_sim3.cpp
  ${PROJECT_SOURCE_DIR}/src/GlobalMapping/c_MapOptimization.cpp
  ${PROJECT_SOURCE_DIR}/src/c_SlamSystem.cpp
)
set(SOURCE_FILES
  ${lsd_SOURCE_FILES}
  # ${PROJECT_SOURCE_DIR}/src/IOWrapper/ROS/ROSImageStreamThread.cpp
)

include_directories(
  ${EIGEN3_INCLUDE_DIR}
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/thirdparty/Sophus
  ${CSPARSE_INCLUDE_DIR} #Has been set by SuiteParse
  ${CHOLMOD_INCLUDE_DIR} #Has been set by SuiteParse
  "D:\\lsd_slam\\g2o-20160424_git"
)


# build shared library.
#rosbuild_add_library(lsdslam SHARED ${SOURCE_FILES})
add_library(lsdslam SHARED ${SOURCE_FILES})
#add_library(lsdslam STATIC ${SOURCE_FILES})
#target_link_libraries(lsdslam g2o_core g2o_stuff csparse cxsparse g2o_solver_csparse g2o_csparse_extension g2o_types_sim3 g2o_types_sba)
#rosbuild_link_boost(lsdslam thread)
target_link_libraries(lsdslam "D:\\lsd_slam\\my_code_2019\\thirdparty\\g2o\\bin\\Debug\\g2o_core_d.lib" "D:\\lsd_slam\\my_code_2019\\thirdparty\\g2o\\bin\\Debug\\g2o_stuff_d.lib"  "D:\\lsd_slam\\my_code_2019\\thirdparty\\g2o\\bin\\Debug\\g2o_solver_csparse_d.lib" "D:\\lsd_slam\\my_code_2019\\thirdparty\\g2o\\bin\\Debug\\g2o_csparse_extension_d.lib" "D:\\lsd_slam\\my_code_2019\\thirdparty\\g2o\\bin\\Debug\\g2o_types_sim3_d.lib" "D:\\lsd_slam\\my_code_2019\\thirdparty\\g2o\\bin\\Debug\\g2o_types_sba_d.lib")
#"D:\\lsd_slam\\suitesparse-metis-for-windows-1.3.0\\build\\lib\\Debug\\libCSparsed.lib" "D:\\lsd_slam\\suitesparse-metis-for-windows-1.3.0\\build\\lib\\Debug\\libcxsparsed.lib" 
# build live ros node
#rosbuild_add_executable(live_slam src/main_live_odometry.cpp)
#target_link_libraries(live_slam lsdslam)


# build image node
#rosbuild_add_executable(dataset_slam src/main_on_images.cpp)

#add_executable(dataset_slam src/main_file.cpp)
#target_link_libraries(dataset_slam lsdslam ${OpenCV_LIBS})

