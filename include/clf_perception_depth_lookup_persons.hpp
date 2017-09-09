// STD
#include <mutex>
#include <cmath>
#include <string>
#include <stdlib.h>

// ROS
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>

// MSGS
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Quaternion.h>
#include <geometry_msgs/Vector3.h>
#include <sensor_msgs/image_encodings.h>
#include <geometry_msgs/TransformStamped.h>
#include <clf_perception_vision/ExtendedPeople.h>
#include <clf_perception_vision/ExtendedPersonStamped.h>

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

std::string people_topic;
std::string depth_topic;
std::string depth_info;
std::string rgb_info;
std::string out_topic;
std::string in_topic;

cv::Mat depth_;
cv::Mat depth_copy;
ros::Time stamp_;
std::string frameId_;
std::mutex im_mutex;
float depthConstant_;
double shift_center_y;

ros::Publisher people_pub;
tf::TransformBroadcaster *tfBroadcaster_;
