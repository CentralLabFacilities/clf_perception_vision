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
#include <clf_perception_vision_msgs/ExtendedPeople.h>
#include <clf_perception_vision_msgs/ExtendedPersonStamped.h>

// FILTER
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>

// CV
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;
using namespace sensor_msgs;
using namespace message_filters;

namespace clf_perception_vision {

    class DepthLookup : public nodelet::Nodelet {
    public:
        DepthLookup() {}

    private:
        ros::NodeHandle private_nh;
        ros::Publisher people_pub;

        string depth_topic;
        string depth_info;
        string out_topic;
        string in_topic;

        virtual void onInit() {
            ROS_INFO("Image Sink Nodelet 'onInit()' done.");
            // Get the NodeHandle
            private_nh = getPrivateNodeHandle();

            ROS_INFO(">>> LOOOOOOL %s", "LOL");

            if (private_nh.getParam("depthlookup_image_topic", depth_topic))
            {
                //NODELET_INFO(">>> Input depth image topic: " << depth_topic);
            } else {
                NODELET_ERROR("!Failed to get depth image topic parameter!");
                exit(EXIT_FAILURE);
            }
            if (private_nh.getParam("depthlookup_info_topic", depth_info))
            {
                //NODELET_INFO(">>> Input depth camera info topic: " << depth_info);
            } else {
                NODELET_ERROR("!Failed to get depth camera info topic parameter!");
                exit(EXIT_FAILURE);
            }

            if (private_nh.getParam("depthlookup_out_topic", out_topic))
            {
                //NODELET_INFO(">>> Output Topic: " << out_topic);
            } else {
                NODELET_ERROR("!Failed to get output topic parameter!");
                exit(EXIT_FAILURE);
            }

            if (private_nh.getParam("depthlookup_in_topic", in_topic))
            {
                //NODELET_INFO(">>> Input Topic: " << in_topic);
            } else {
                NODELET_ERROR("!Failed to get input topic parameter!");
                exit(EXIT_FAILURE);
            }

            message_filters::Subscriber<Image> image_sub(private_nh, depth_topic.c_str(), 5);
            message_filters::Subscriber<CameraInfo> info_sub(private_nh, depth_info.c_str(), 5);
            people_pub = private_nh.advertise<clf_perception_vision::ExtendedPeople>(out_topic.c_str(), 5);

            TimeSynchronizer<Image, CameraInfo> sync(image_sub, info_sub, 5);
            sync.registerCallback(boost::bind(&DepthLookup::depth_callback, this ,_1, _2));
        }

        void depth_callback(const ImageConstPtr& image, const CameraInfoConstPtr& cam_info) {
            ROS_INFO(">>> callback");
        }

    };

    PLUGINLIB_DECLARE_CLASS(clf_perception_vision, DepthLookup, clf_perception_vision::DepthLookup, nodelet::Nodelet);
}
