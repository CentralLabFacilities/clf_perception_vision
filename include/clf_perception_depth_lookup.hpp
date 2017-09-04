// STD
#include <stdlib.h>
#include <cmath>

// ROS
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>

// MSGS
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/image_encodings.h>
#include <clf_perception_vision/ExtenedPeople.h>
#include <clf_perception_vision/ExtendedPersonStamped.h>

// FILTER
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>

// CV
#include <opencv2/imgproc/imgproc.hpp>

// TF
#include <tf/transform_broadcaster.h>

std::string people_topic;
std::string depth_topic;
std::string depth_info;
std::string out_topic;
std::string in_topic;

std::string frameId_;
ros::Time stamp_;
cv::Mat depth_;
float depthConstant_;