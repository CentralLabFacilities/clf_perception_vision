// STD
#include <stdlib.h>

// ROS NODELETS
#include <ros/ros.h>
#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>

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

using namespace cv;
using namespace std;
using namespace sensor_msgs;
using namespace message_filters;

void depthCallback(const ImageConstPtr& image, const CameraInfoConstPtr& cam_info)
{
  ROS_INFO("callback");
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "clf_perception_depth_lookup", ros::init_options::AnonymousName);
    ros::Publisher people_pub;
    ros::NodeHandle nh;

    string depth_topic;
    string depth_info;
    string out_topic;
    string in_topic;

    if (nh.getParam("depthlookup_image_topic", depth_topic))
    {
        ROS_INFO(">>> Input depth image topic: %s", depth_topic.c_str());
    } else {
        ROS_ERROR("!Failed to get depth image topic parameter!");
        exit(EXIT_FAILURE);
    }
    if (nh.getParam("depthlookup_info_topic", depth_info))
    {
        ROS_INFO(">>> Input depth camera info topic: %s", depth_info.c_str());
    } else {
        ROS_ERROR("!Failed to get depth camera info topic parameter!");
        exit(EXIT_FAILURE);
    }

    if (nh.getParam("depthlookup_out_topic", out_topic))
    {
        ROS_INFO(">>> >>> Output Topic:  %s", out_topic.c_str());
    } else {
        ROS_ERROR("!Failed to get output topic parameter!");
        exit(EXIT_FAILURE);
    }

    if (nh.getParam("depthlookup_in_topic", in_topic))
    {
        ROS_INFO(">>> Input Topic:  %s", in_topic.c_str());
    } else {
        ROS_ERROR("!Failed to get input topic parameter!");
        exit(EXIT_FAILURE);
    }

    message_filters::Subscriber<Image> image_sub(nh, depth_topic, 5);
    message_filters::Subscriber<CameraInfo> info_sub(nh, depth_info, 5);
    people_pub = nh.advertise<clf_perception_vision::ExtenedPeople>(out_topic, 5);

    TimeSynchronizer<Image, CameraInfo> sync(image_sub, info_sub, 5);
    sync.registerCallback(boost::bind(depthCallback, _1, _2));

    ros::spin();
    return 0;
}