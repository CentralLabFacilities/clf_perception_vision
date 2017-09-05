// STD
#include <cmath>
#include <stdlib.h>
#include <string>

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
#include <clf_perception_vision/ExtendedObjects.h>
#include <clf_perception_vision/ExtendedObjectsStamped.h>

// FILTER
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

// CV
#include <opencv2/imgproc/imgproc.hpp>

// TF
#include <tf/transform_broadcaster.h>

std::string objects_topic;
std::string depth_topic;
std::string depth_info;
std::string rgb_info;
std::string out_topic;
std::string in_topic;

cv::Mat depth_;
ros::Time stamp_;
std::string frameId_;
float depthConstant_;
double shift_center_y;

ros::Publisher objects_pub;
tf::TransformBroadcaster *tfBroadcaster_;
