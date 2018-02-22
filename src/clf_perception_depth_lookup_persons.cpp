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

#include "clf_perception_depth_lookup_persons.hpp"

using namespace cv;
using namespace std;
using namespace sensor_msgs;
using namespace geometry_msgs;
using namespace message_filters;
using namespace clf_perception_vision_msgs;

// This piece of code luckily already existed in https://github.com/introlab/find-object/blob/master/src/ros/FindObjectROS.cpp (THANKS!)
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
	bool isValid;

	if(isInMM) {
	    // ROS_DEBUG(">>> Image is in Millimeters");
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

        depth = median;
		ROS_DEBUG("%f", depth);
		isValid = depth != 0.0f;

	} else {
		// ROS_DEBUG(">>> Image is in Meters");
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

        depth = median;
        ROS_DEBUG("%f", depth);
		isValid = isfinite(depth);
	}

	// Check for invalid measurements
	if (!isValid)
	{
	    ROS_DEBUG(">>> WARN Image is invalid, whoopsie.");
		pt.val[0] = pt.val[1] = pt.val[2] = bad_point;
	} else{
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
    }
}

void rgbInfoCallback(const CameraInfoConstPtr& cameraInfoMsgRgb) {
    if(!camera_image_rgb_width_is_set) {
        ROS_INFO(">>> Setting camera_rgb_width");
        camera_image_rgb_width = cameraInfoMsgRgb->width;
        camera_image_rgb_width_is_set = true;
    }
}

void syncCallback(const ImageConstPtr& depthMsg, const ImageConstPtr& colorMsg, const ExtendedPeopleConstPtr& peopleMsg) {

    if(!depthConstant_factor_is_set) {
        ROS_WARN(">>> Waiting for first depth camera info message to arrive...");
        return;
    }

    if(!camera_image_rgb_width_is_set) {
        ROS_WARN(">>> Waiting for first rgb camera info message to arrive...");
        return;
    }

    // Lock image
    im_mutex.lock();

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
      ROS_ERROR(">>> cv_bridge exception: %s", e.what());
      return;
    }

    float depthConstant = 1.0f/depthConstant_factor;

    setDepthData(depthMsg->header.frame_id, depthMsg->header.stamp, ptrDepth->image, depthConstant);

    cv_bridge::CvImageConstPtr ptrColor;

    try {
      ptrColor = cv_bridge::toCvCopy(colorMsg, sensor_msgs::image_encodings::BGR8);
    } catch (cv_bridge::Exception& e) {
      ROS_ERROR(">>> cv_bridge exception: %s", e.what());
      return;
    }

    // Create a new image to process below
    cv::Mat im = ptrColor->image;

    // Copy Message in order to manipulate it later and sent updated version.
    ExtendedPeople people_cpy;
    people_cpy = *peopleMsg;

    vector<tf::StampedTransform> transforms;
    vector<cv::Rect> rectangles;
    vector<cv::Point> points;
    vector<std::string> probabilities;

    // Common time. Copied from extended people, which
    // in turn has been copied from the time of detection
    // using Darknet Yolo
    ros::Time current_stamp = people_cpy.header.stamp;

    // Pose Array of people
    PoseArray pose_arr;
    pose_arr.header.stamp = current_stamp;
    pose_arr.header.frame_id = frameId_;

    // Pose extended msgs
    ExtendedPoseArray pose_ex;
    // Add header to PoseArrayExtended
    pose_ex.header = people_cpy.header;

    int bbox_xmin, bbox_xmax, bbox_ymin, bbox_ymax;

    // If depth image and color image have different resolutions,
    // derive a factor to scale the bounding boxes
    float scale_factor = camera_image_rgb_width/camera_image_depth_width;

    ROS_DEBUG(">>> Scale ratio RGB Image to DEPTH image is: %f ", scale_factor);

    for (int i=0; i<peopleMsg->persons.size(); i++) {
        bbox_xmin = peopleMsg->persons[i].bbox_xmin;
        bbox_xmax = peopleMsg->persons[i].bbox_xmax;
        bbox_ymin = peopleMsg->persons[i].bbox_ymin;
        bbox_ymax = peopleMsg->persons[i].bbox_ymax;
        string probability = to_string(peopleMsg->persons[i].probability);

        float objectWidth = (bbox_xmax - bbox_xmin) / scale_factor;
        float objectHeight = (bbox_ymax - bbox_ymin) / scale_factor;
        float center_x = (bbox_xmin + bbox_xmax) / scale_factor / 2;
        float center_y = ( (bbox_ymin + bbox_ymax) / scale_factor / shift_center_y ) / 2;

        // float xAxis_x = 3*objectWidth/4;
        // float xAxis_y = objectHeight/2;
        // float yAxis_x = objectWidth/2;
        // float yAxis_y = 3*objectHeight/4;

        cv::Vec3f center3D = getDepth(depth_,
				center_x+0.5f, center_y+0.5f,
				float(depth_.cols/2)-0.5f, float(depth_.rows/2)-0.5f,
				1.0f/depthConstant_, 1.0f/depthConstant_);

        //cv::Vec3f axisEndX = getDepth(depth_,
        //         xAxis_x+0.5f, xAxis_y+0.5f,
        //        float(depth_.cols/2)-0.5f, float(depth_.rows/2)-0.5f,
        //        1.0f/depthConstant_, 1.0f/depthConstant_);

        //cv::Vec3f axisEndY = getDepth(depth_,
        //        yAxis_x+0.5f, yAxis_y+0.5f,
        //        float(depth_.cols/2)-0.5f, float(depth_.rows/2)-0.5f,
        //        1.0f/depthConstant_, 1.0f/depthConstant_);

        string id = "person_" + to_string(i);

        if (isfinite(center3D.val[0]) && isfinite(center3D.val[1]) && isfinite(center3D.val[2])) {

            // Setup a rectangle to define your region of interest
            cv::Rect roi(bbox_xmin, bbox_ymin, objectWidth * scale_factor, objectHeight * scale_factor);
            rectangles.push_back(roi);
            points.push_back(cv::Point(center_x * scale_factor, center_y * scale_factor));
            probabilities.push_back(probability);

            // Crop the full image to that image contained by the rectangle roi
            // Note that this doesn't copy the data!
            cv::Mat croppedImage = im(roi);

            // Compose image message
            cv_bridge::CvImage image_out_msg;
            image_out_msg.header   = people_cpy.header;
            image_out_msg.encoding = sensor_msgs::image_encodings::BGR8;
            image_out_msg.image    = croppedImage;
            pose_ex.images.push_back(*image_out_msg.toImageMsg());

            tf::StampedTransform transform;
            transform.setIdentity();
            transform.child_frame_id_ = id;
            transform.frame_id_ = frameId_;
            transform.stamp_ = current_stamp;
            transform.setOrigin(tf::Vector3(center3D.val[0], center3D.val[1], center3D.val[2]));

            // set rotation (y inverted)
            // tf::Vector3 xAxis(axisEndX.val[0] - center3D.val[0], axisEndX.val[1] - center3D.val[1], axisEndX.val[2] - center3D.val[2]);
            // xAxis.normalize();
            // tf::Vector3 yAxis(axisEndY.val[0] - center3D.val[0], axisEndY.val[1] - center3D.val[1], axisEndY.val[2] - center3D.val[2]);
            // yAxis.normalize();
            // tf::Vector3 zAxis = xAxis*yAxis;
            // tf::Matrix3x3 rotationMatrix(
            //            xAxis.x(), yAxis.x(), zAxis.x(),
            //            xAxis.y(), yAxis.y(), zAxis.y(),
            //            xAxis.z(), yAxis.z(), zAxis.z());
            // tf::Quaternion q;
            // rotationMatrix.getRotation(q);
            // set x axis going front of the object, with z up and z left
            // q *= tf::createQuaternionFromRPY(CV_PI/2.0, CV_PI/2.0, 0);
            // transform.setRotation(q.normalized());

            PoseStamped pose_stamped;
            pose_stamped.header = people_cpy.header;
            pose_stamped.header.frame_id = frameId_;
            pose_stamped.pose.position.x = center3D.val[0];
            pose_stamped.pose.position.y = center3D.val[1];
            pose_stamped.pose.position.z = center3D.val[2];
            pose_stamped.pose.orientation.x = 0.0; //q.normalized().x();
            pose_stamped.pose.orientation.y = 0.0; //q.normalized().y();
            pose_stamped.pose.orientation.z = 0.0; //q.normalized().z();
            pose_stamped.pose.orientation.w = 1.0; //q.normalized().w();

            Pose pose;
            pose.position.x = center3D.val[0];
            pose.position.y = center3D.val[1];
            pose.position.z = center3D.val[2];
            pose.orientation.x = 0.0; //q.normalized().x();
            pose.orientation.y = 0.0; //q.normalized().y();
            pose.orientation.z = 0.0; //q.normalized().z();
            pose.orientation.w = 1.0; //q.normalized().w();

            ///////// FILL ///////////////////////////////////////////////
            people_cpy.persons[i].pose = pose_stamped;
            people_cpy.persons[i].transformid = id;
            transforms.push_back(transform);
            pose_arr.poses.push_back(pose);

            ROS_DEBUG(">>> person_%d detected, center 2D at (%f,%f) setting frame \"%s\" \n", i, center_x, center_y, id.c_str());
		} else {
			ROS_DEBUG(">>> WARN person_%d detected, center 2D at (%f,%f), but invalid depth, cannot set frame \"%s\"!\n", i, center_x, center_y, id.c_str());
		}
    }

    // Fill pose array
    pose_ex.poses = pose_arr;

    // Finally unlock
    im_mutex.unlock();

    if(transforms.size() > 0) {
   	   tfBroadcaster_->sendTransform(transforms);
	   people_pub.publish(people_cpy);
       people_pub_pose.publish(pose_arr);
       people_pub_extended_pose.publish(pose_ex);
    }

    for(int i=0; i<rectangles.size(); i++) {
        cv::circle(im, points[i], 10, Scalar(207, 161, 88), CV_FILLED);
        cv::rectangle(im, rectangles[i], Scalar(0, 255, 255), 2, 8, 0);
        cv::putText(im, "p "+probabilities[i].substr(0,4), cv::Point(points[i].x+12, points[i].y+5 ), fontFace, fontScale, cv::Scalar(207, 161, 88), 1);
    }

    cv::imshow("CLF PERCEPTION || Depth Lookup", im);
    cv::waitKey(1);
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "clf_perception_depth_lookup", ros::init_options::AnonymousName);
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

    if (nh.getParam("depthlookup_in_topic", in_topic))
    {
        ROS_INFO(">>> Input Topic: %s", in_topic.c_str());
    } else {
        ROS_ERROR("!Failed to get input topic parameter!");
        exit(EXIT_FAILURE);
    }

    if (nh.getParam("depthlookup_out_topic", out_topic))
    {
        ROS_INFO(">>> Output Topic: %s", out_topic.c_str());
    } else {
        ROS_ERROR("!Failed to get output topic parameter!");
        exit(EXIT_FAILURE);
    }

    if (nh.getParam("depthlookup_out_topic_pose", out_topic_pose))
    {
        ROS_INFO(">>> Output Topic Pose: %s", out_topic_pose.c_str());
    } else {
        ROS_ERROR("!Failed to get pose output topic parameter!");
        exit(EXIT_FAILURE);
    }

    if (nh.getParam("depthlookup_out_topic_pose_extended", out_topic_pose_extended))
    {
        ROS_INFO(">>> Output Topic Pose Extended: %s", out_topic_pose_extended.c_str());
    } else {
        ROS_ERROR("!Failed to get pose extended output topic parameter!");
        exit(EXIT_FAILURE);
    }

    if (nh.getParam("depthlookup_shift_center_y", shift_center_y))
    {
        ROS_INFO(">>> Shift center_y: %f", shift_center_y);
    } else {
        ROS_ERROR("!Failed to get output topic parameter!");
        exit(EXIT_FAILURE);
    }

    tfBroadcaster_ = new tf::TransformBroadcaster();

    info_depth_sub = nh.subscribe(depth_info, 2, depthInfoCallback);
    info_rgb_sub = nh.subscribe(rgb_info, 2, rgbInfoCallback);

    Subscriber<Image> depth_image_sub(nh, depth_topic, 1);
    Subscriber<Image> rgb_image_sub(nh, rgb_topic, 1);

    Subscriber<ExtendedPeople> people_sub(nh, in_topic, 1);

    typedef sync_policies::ApproximateTime<Image, Image ,ExtendedPeople> sync_pol;

    Synchronizer<sync_pol> sync(sync_pol(10), depth_image_sub, rgb_image_sub, people_sub);
    sync.registerCallback(boost::bind(&syncCallback, _1, _2, _3));

    people_pub = nh.advertise<ExtendedPeople>(out_topic, 1);
    people_pub_pose = nh.advertise<PoseArray>(out_topic_pose, 1);
    people_pub_extended_pose = nh.advertise<ExtendedPoseArray>(out_topic_pose_extended, 1);

    cv::namedWindow("CLF PERCEPTION || Depth Lookup", WINDOW_AUTOSIZE);

    ros::spin();

    return 0;
}