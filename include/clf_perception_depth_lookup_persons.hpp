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

// STD
#include <mutex>
#include <cmath>
#include <string>
#include <limits>
#include <stdlib.h>

// ROS
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>

// MSGS

#include <sensor_msgs/Image.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/Vector3.h>
#include <sensor_msgs/CameraInfo.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Quaternion.h>
#include <geometry_msgs/PoseArray.h>
#include <sensor_msgs/image_encodings.h>
#include <geometry_msgs/TransformStamped.h>
#include <clf_perception_vision_msgs/ExtendedPeople.h>
#include <clf_perception_vision_msgs/ExtendedPoseArray.h>
#include <clf_perception_vision_msgs/ExtendedPersonStamped.h>

// FILTER
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

// CV
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// TF
#include <tf/transform_broadcaster.h>

std::string out_topic_pose_extended;
std::string out_topic_pose;
std::string people_topic;
std::string depth_topic;
std::string depth_info;
std::string rgb_topic;
std::string out_topic;
std::string rgb_info;
std::string in_topic;

cv::Mat depth_;
cv::Mat depth_copy;
ros::Time stamp_;
std::string frameId_;
std::mutex im_mutex;

float depthConstant_;
float depthConstant_factor;
float camera_image_rgb_width;
float camera_image_depth_width;
double shift_center_y;

bool depthConstant_factor_is_set = false;
bool camera_image_rgb_width_is_set = false;

ros::Publisher people_pub;
ros::Publisher people_pub_pose;
ros::Publisher people_pub_extended_pose;

ros::Subscriber info_depth_sub;
ros::Subscriber info_rgb_sub;

tf::TransformBroadcaster *tfBroadcaster_;


const int fontFace = cv::FONT_HERSHEY_PLAIN;
const double fontScale = 1;