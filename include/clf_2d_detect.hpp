/*

Author: Florian Lier [flier AT techfak.uni-bielefeld DOT de]

By downloading, copying, installing or using the software you agree to this license.
If you do not agree to this license, do not download, install, copy or use the software.

                          License Agreement
               For Open Source Computer Vision Library
                       (3-clause BSD License)

Copyright (C) 2000-2016, Intel Corporation, all rights reserved.
Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
Copyright (C) 2009-2016, NVIDIA Corporation, all rights reserved.
Copyright (C) 2010-2013, Advanced Micro Devices, Inc., all rights reserved.
Copyright (C) 2015-2016, OpenCV Foundation, all rights reserved.
Copyright (C) 2015-2016, Itseez Inc., all rights reserved.
Third party copyrights are property of their respective owners.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

  * Neither the names of the copyright holders nor the names of the contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.

This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are disclaimed.
In no event shall copyright holders or contributors be liable for any direct,
indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.

*/

#pragma once

// STD
#include <vector>
#include <time.h>
#include <string>
#include <iostream>

// OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

// CUDA
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/xfeatures2d/cuda.hpp>

// BOOST
#include <boost/date_time/posix_time/posix_time.hpp>

// ROS
#include <ros/ros.h>
#include <visualization_msgs/InteractiveMarkerPose.h>
#include <std_msgs/Header.h>
#include <geometry_msgs/Point.h>

class Detect2D {

public:
    int setup(int argc, char *argv[]);
    void detect(cv::Mat i_image, std::string capture_duration, ros::Time timestamp);
    std::vector<cv::Scalar> color_mix();
    int get_x_resolution();
    int get_y_resolution();
    bool get_silent();
    Detect2D();
    ~Detect2D();

    const int fontFace = cv::FONT_HERSHEY_PLAIN;
    const double fontScale = 1;

private:
    std::vector<cv::Scalar> colors;
    std::vector<std::string> target_paths;
    std::vector<std::string> target_labels;
    std::vector<cv::Mat> target_images;
    std::vector<std::vector<cv::KeyPoint>> keys_current_target;
    std::vector<cv::cuda::GpuMat> cuda_desc_current_target_image;
    std::vector<cv::KeyPoint> keys_camera_image;
    cv::Point2d *target_medians;

    const int text_origin = 10;
    int max_keypoints = 0;
    int max_number_matching_points = 0;
    int text_offset_y = 20;
    int detection_threshold = 0;
    int res_x = 640;
    int res_y = 480;
    double scale_factor = 1.0;

    bool do_not_draw = false;
    bool toggle_homography = false;
    bool toggle_silent = false;

    std::string type_descriptor;
    std::string point_matcher;
    std::string draw_homography;
    std::string draw_image;

    cv::cuda::GpuMat cuda_frame_scaled;
    cv::cuda::GpuMat cuda_camera_tmp_img;
    cv::cuda::GpuMat cuda_desc_camera_image;

    cv::Ptr<cuda::ORB> cuda_orb;
    cv::Ptr<cv::cuda::DescriptorMatcher> cuda_bf_matcher;

    ros::NodeHandle node_handle_;
    ros::Publisher object_pub;
    visualization_msgs::InteractiveMarkerPose msg;
    std_msgs::Header h;
    geometry_msgs::Point pt;
};
