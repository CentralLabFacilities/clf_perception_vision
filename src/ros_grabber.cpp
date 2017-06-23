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

// SELF
#include "ros_grabber.hpp"


ROSGrabber::ROSGrabber(std::string i_scope) : it_(node_handle_) {
    image_sub_ = it_.subscribe(i_scope, 100, &ROSGrabber::imageCallback, this);
    last_frame = ros::Time::now();
}

ROSGrabber::~ROSGrabber() { }

void ROSGrabber::imageCallback(const sensor_msgs::ImageConstPtr &msg) {
    cv_bridge::CvImagePtr cv_ptr;

    try {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception &e) {
        ROS_ERROR("E >>> CV_BRIDGE exception: %s", e.what());
        return;
    }

    mtx.lock();
    ros::Time frame_time = msg->header.stamp;
    frame_nr = (int)msg->header.seq;
    timestamp = frame_time;
    source_frame = cv_ptr->image;
    output_frame = source_frame;
    mtx.unlock();
}

void ROSGrabber::getImage(cv::Mat *mat) {
    mtx.lock();
    *mat = output_frame;
    last_frame = timestamp;
    mtx.unlock();
}

ros::Time ROSGrabber::getTimestamp() {
    return last_frame;
}

int ROSGrabber::getLastFrameNr() {
    return frame_nr;
}