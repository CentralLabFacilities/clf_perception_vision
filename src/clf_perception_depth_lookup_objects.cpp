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

#include "clf_perception_depth_lookup_objects.hpp"

using namespace cv;
using namespace std;
using namespace sensor_msgs;
using namespace geometry_msgs;
using namespace message_filters;
using namespace object_tracking_msgs;

// This function luckily already existed in https://github.com/introlab/find-object/blob/master/src/ros/FindObjectROS.cpp (THANKS!)
Vec3f getDepth(const Mat & depthImage, int x, int y, float cx, float cy, float fx, float fy) {
	if(!(x >=0 && x<depthImage.cols && y >=0 && y<depthImage.rows))
	{
		ROS_ERROR(">>> Point must be inside the image (x=%d, y=%d), image size=(%d,%d)", x, y, depthImage.cols, depthImage.rows);
		return Vec3f(
				numeric_limits<float>::quiet_NaN(),
				numeric_limits<float>::quiet_NaN(),
				numeric_limits<float>::quiet_NaN());
	}

	cv::Vec3f pt;

	// Use correct principal point from calibration
	float center_x = cx; //cameraInfo.K.at(2)
	float center_y = cy; //cameraInfo.K.at(5)

	bool isInMM = depthImage.type() == CV_16UC1; // is in mm?

	// Combine unit conversion (if necessary) with scaling by focal length for computing (X,Y)
	float unit_scaling = isInMM?0.001f:1.0f;
	float constant_x = unit_scaling / fx; //cameraInfo.K.at(0)
	float constant_y = unit_scaling / fy; //cameraInfo.K.at(4)
	float bad_point = numeric_limits<float>::quiet_NaN();

	float depth;
    //minimum depth of an object roi (in m)
    float minDepth = 0.2;
	bool isValid;

	if(isInMM) {
        ROS_DEBUG(">>> Image is in Millimeters");
	    float depth_samples[21];

        // Sample fore depth points to the right, left, top and down
        for (int i=0; i<5; i++) {
            depth_samples[i] = (float)depthImage.at<uint16_t>(y,x+i);
            depth_samples[i+5] = (float)depthImage.at<uint16_t>(y,x-i);
            depth_samples[i+10] = (float)depthImage.at<uint16_t>(y+i,x);
            depth_samples[i+15] = (float)depthImage.at<uint16_t>(y-i,x);
        }

        depth_samples[20] = (float)depthImage.at<uint16_t>(y, x);

        int arr_size = sizeof(depth_samples)/sizeof(float);
        sort(&depth_samples[0], &depth_samples[arr_size]);
        float median = arr_size % 2 ? depth_samples[arr_size/2] : (depth_samples[arr_size/2-1] + depth_samples[arr_size/2]) / 2;
        // get roi depth (at least  minDepth)
        float min = 9999;
        float max = 0.0;
        for(int i = 0; i < arr_size; i++) {
            if(depth_samples[i] < min) {
                min = depth_samples[i];
            }
            if(depth_samples[i] > max) {
                max = depth_samples[i];
            }
        }
        objectDepth = max - min;
        if (objectDepth < minDepth*1000) {
            objectDepth = minDepth*1000;
        }
        ROS_DEBUG("roi depth: %f", objectDepth);
        ROS_DEBUG("convert roi depth to m");
        objectDepth = objectDepth/1000;

        depth = median;
        ROS_DEBUG("depth: %f", depth);
        isValid = depth != 0.0f;

	} else {
        ROS_DEBUG(">>> Image is in Meters");
		float depth_samples[21];

        // Sample fore depth points to the right, left, top and down
        for (int i=0; i<5; i++) {
            depth_samples[i] = depthImage.at<float>(y,x+i);
            depth_samples[i+5] = depthImage.at<float>(y,x-i);
            depth_samples[i+10] = depthImage.at<float>(y+i,x);
            depth_samples[i+15] = depthImage.at<float>(y-i,x);
        }

        depth_samples[20] = depthImage.at<float>(y,x);

        int arr_size = sizeof(depth_samples)/sizeof(float);
        sort(&depth_samples[0], &depth_samples[arr_size]);
        float median = arr_size % 2 ? depth_samples[arr_size/2] : (depth_samples[arr_size/2-1] + depth_samples[arr_size/2]) / 2;

        // get roi depth (at least 0.3m)
        float min = 9.9;
        float max = 0.0;
        for(int i = 0; i < arr_size; i++) {
            if(depth_samples[i] < min) {
                min = depth_samples[i];
            }
            if(depth_samples[i] > max) {
                max = depth_samples[i];
            }
        }
        objectDepth = max - min;
        if (objectDepth < minDepth) {
            objectDepth = minDepth;
        }
        ROS_INFO("roi depth: %f", objectDepth);

        depth = median;
        ROS_DEBUG("depth: %f", depth);
        isValid = isfinite(depth);
	}

	// Check for invalid measurements
	if (!isValid)
	{
        ROS_INFO(">>> WARN Image is invalid, whoopsie.");
		pt.val[0] = pt.val[1] = pt.val[2] = bad_point;
	} else{
        ROS_INFO(">>> Image is valid.");
		// Fill in XYZ
		pt.val[0] = (float(x) - center_x) * depth * constant_x;
		pt.val[1] = (float(y) - center_y) * depth * constant_y;
		pt.val[2] = depth*unit_scaling;
	}

	return pt;
}

void setDepthData(const string &frameId, const ros::Time &stamp, const Mat &depth, float depthConstant) {
    frameId_ = frameId;
    stamp_ = stamp;
    depth_ = depth;
    depthConstant_ = depthConstant;
}

void depthInfoCallback(const CameraInfoConstPtr& cameraInfoMsg) {
    if(!depthConstant_factor_is_set) {
        ROS_INFO(">>> Setting depthConstant_factor");
        depthConstant_factor = cameraInfoMsg->K[4];
        camera_image_depth_width = cameraInfoMsg->width;
        depthConstant_factor_is_set = true;
    } else {
      // Unsubscribe, we only need that once.
      info_depth_sub.shutdown();
    }
}

void rgbInfoCallback(const CameraInfoConstPtr& cameraInfoMsgRgb) {
    if(!camera_image_rgb_width_is_set) {
        ROS_INFO(">>> Setting camera_rgb_width");
        camera_image_rgb_width = cameraInfoMsgRgb->width;
        camera_image_rgb_width_is_set = true;
    } else {
      // Unsubscribe, we only need that once.
      info_rgb_sub.shutdown();
    }
}

void syncCallback(const ImageConstPtr& depthMsg, const ImageConstPtr& colorMsg) {

    if(!depthConstant_factor_is_set) {
        ROS_WARN(">>> Waiting for first depth camera INFO message to arrive...");
        return;
    }

    if(!camera_image_rgb_width_is_set) {
        ROS_WARN(">>> Waiting for first rgb camera INFO message to arrive...");
        return;
    }

    ///////////////////////////// Image conversion ////////////////////////////////////////////

    cv_bridge::CvImageConstPtr ptrDepth;

    try {
        if (depthMsg->encoding == "16UC1") {
           ptrDepth = cv_bridge::toCvShare(depthMsg, sensor_msgs::image_encodings::TYPE_16UC1);
        } else if (depthMsg->encoding == "32FC1") {
           ptrDepth = cv_bridge::toCvShare(depthMsg, sensor_msgs::image_encodings::TYPE_32FC1);
        } else {
          ROS_ERROR(">>> Unknown image encoding %s", depthMsg->encoding.c_str());
          im_mutex.unlock();
          return;
        }
    } catch (cv_bridge::Exception& e) {
      ROS_ERROR(">>> CV_BRIDGE exception: %s", e.what());
      return;
    }

    cv_bridge::CvImageConstPtr ptrColor;

    try {
      ptrColor = cv_bridge::toCvCopy(colorMsg, sensor_msgs::image_encodings::BGR8);
    } catch (cv_bridge::Exception& e) {
      ROS_ERROR(">>> CV_BRIDGE exception: %s", e.what());
      return;
    }

    ///////////////////////////// End image conversion /////////////////////////////////////////

    float depthConstant = 1.0f/depthConstant_factor;
    setDepthData(depthMsg->header.frame_id, depthMsg->header.stamp, ptrDepth->image, depthConstant);

    // Create a new image to process below
    cv::Mat im = ptrColor->image;
    cv::Mat im_depth = ptrDepth->image;

}

bool srvCallback(object_tracking_msgs::DepthLookup::Request &req, object_tracking_msgs::DepthLookup::Response &res) {

    // check if depth and rgb data has arrived yet
    if(!depthConstant_factor_is_set) {
        ROS_WARN(">>> Waiting for first depth camera INFO message to arrive...");
        return false;
    }
    if(!camera_image_rgb_width_is_set) {
        ROS_WARN(">>> Waiting for first rgb camera INFO message to arrive...");
        return false;
    }

    vector<cv::Point> points;
    vector<cv::Rect> rectangles;
    vector<tf::StampedTransform> transforms;
    vector<std::string> probabilities;

    int bbox_xmin, bbox_xmax, bbox_ymin, bbox_ymax;

    // If depth image and color image have different resolutions,
    // derive factor to scale the bounding boxes
    float scale_factor = camera_image_rgb_width/camera_image_depth_width;
    ROS_DEBUG(">>> Scale ratio RGB --> DEPTH image is: %f ", scale_factor);

    for (int i=0; i<req.objectLocationList.size(); i++) {

        bbox_xmin = req.objectLocationList[i].bounding_box.x_offset;
        bbox_ymin = req.objectLocationList[i].bounding_box.y_offset;
        bbox_xmax = req.objectLocationList[i].bounding_box.width + bbox_xmin;
        bbox_ymax = req.objectLocationList[i].bounding_box.height + bbox_ymin;

        float objectWidth = (bbox_xmax - bbox_xmin) / scale_factor;
        float objectHeight = (bbox_ymax - bbox_ymin) / scale_factor;
        float center_x = (bbox_xmin + bbox_xmax) / scale_factor / 2;
        float center_y = ( (bbox_ymin + bbox_ymax) / scale_factor / shift_center_y ) / 2;

        cv::Vec3f center3D = getDepth(depth_,
				center_x+0.5f, center_y+0.5f,
				float(depth_.cols/2)-0.5f, float(depth_.rows/2)-0.5f,
				1.0f/depthConstant_, 1.0f/depthConstant_);
        //TODO: convert to 3d roi
        if (isfinite(center3D.val[0]) && isfinite(center3D.val[1]) && isfinite(center3D.val[2])) {

            float depth = 0.2;
            if (objectDepth != 0.0f) {
                depth = objectDepth;
            }

            object_tracking_msgs::ObjectShape objectShape;
            ROS_INFO(">>> got point: (%f,%f,%f)", center3D.val[0], center3D.val[1], center3D.val[2]);
            // write 3d pose to objectShape
            objectShape.center.x = center3D.val[0];
            objectShape.center.y = center3D.val[1];
            objectShape.center.z = center3D.val[2];
            //depth value is just an assumption (at least 0.2m, should be in m!)
            //TODO: get width and height (and depth?) according to 2d roi size
            objectShape.width = depth;
            objectShape.height = depth;
            objectShape.depth = depth;

            // fill objectShape with data from objectLocation
            objectShape.bounding_box = req.objectLocationList[i].bounding_box;
            objectShape.hypotheses = req.objectLocationList[i].hypotheses;
            objectShape.name = req.objectLocationList[i].name;

            res.objectShapeList.push_back(objectShape);
        }
    }
    return true;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "clf_perception_depth_lookup_objects", ros::init_options::AnonymousName);
    ros::NodeHandle nh("~");

    if (nh.getParam("depthlookup_image_topic", depth_topic))
    {
        ROS_INFO(">>> Input depth image topic: %s", depth_topic.c_str());
    } else {
        ROS_ERROR("!Failed to get depth image topic parameter!");
        exit(EXIT_FAILURE);
    }

    if (nh.getParam("depthlookup_image_rgb_topic", rgb_topic))
    {
        ROS_INFO(">>> Input rgb image topic: %s", rgb_topic.c_str());
    } else {
        ROS_ERROR("!Failed to get rgb image topic parameter!");
        exit(EXIT_FAILURE);
    }

    if (nh.getParam("depthlookup_depth_info_topic", depth_info))
    {
        ROS_INFO(">>> Input depth camera info topic: %s", depth_info.c_str());
    } else {
        ROS_ERROR("!Failed to get depth camera info topic parameter!");
        exit(EXIT_FAILURE);
    }

    if (nh.getParam("depthlookup_rgb_info_topic", rgb_info))
    {
        ROS_INFO(">>> Input rgb camera info topic: %s", rgb_info.c_str());
    } else {
        ROS_ERROR("!Failed to get rgb camera info topic parameter!");
        exit(EXIT_FAILURE);
    }

    if (nh.getParam("depthlookup_shift_center_y", shift_center_y))
    {
        ROS_INFO(">>> Shift center_y: %f", shift_center_y);
    } else {
        ROS_ERROR("!Failed to get output topic parameter!");
        exit(EXIT_FAILURE);
    }

    // Subscriber for camera info topics
    info_depth_sub = nh.subscribe(depth_info, 2, depthInfoCallback);
    info_rgb_sub = nh.subscribe(rgb_info, 2, rgbInfoCallback);

    // Subscriber for depth can rgb images
    Subscriber<Image> depth_image_sub(nh, depth_topic, 2);
    Subscriber<Image> rgb_image_sub(nh, rgb_topic, 2);

    // Synchronize depth and rgb input
    typedef sync_policies::ApproximateTime<Image, Image> sync_pol;
    Synchronizer<sync_pol> sync(sync_pol(10), depth_image_sub, rgb_image_sub);
    sync.registerCallback(boost::bind(&syncCallback, _1, _2));

    ros::ServiceServer service = nh.advertiseService("depthLookup", srvCallback);

    ros::spin();

    return 0;
}
