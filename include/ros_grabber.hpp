#pragma once

// ROS
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>

// STD
#include <string>
#include <iostream>
#include <sstream>
#include <mutex>

// CV
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>


// BOOST
#include "boost/date_time/posix_time/posix_time.hpp"

class ROSGrabber {

public:
    ROSGrabber(std::string i_scope);
    ~ROSGrabber();
    void imageCallback(const sensor_msgs::ImageConstPtr& msg);
    void getImage(cv::Mat *mat);
    ros::Time getTimestamp();
    std::string getDuration();
private:
    ros::NodeHandle node_handle_;
    image_transport::ImageTransport it_;
    image_transport::Subscriber image_sub_;
    cv::Mat output_frame;
    cv::Mat source_frame;
    ros::Time timestamp;
    ros::Time last_frame;
    std::string duration;
    std::recursive_mutex mtx;
};

