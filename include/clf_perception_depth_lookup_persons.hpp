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

std::string out_topic_pose;
std::string people_topic;
std::string depth_topic;
std::string depth_info;
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
ros::Subscriber info_depth_sub;
ros::Subscriber info_rgb_sub;
tf::TransformBroadcaster *tfBroadcaster_;
