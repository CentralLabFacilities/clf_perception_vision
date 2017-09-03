// ROS NODELETS
#include <ros/ros.h>
#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>

// MSGS
#include <clf_perception_vision/ExtenedPeople.h>
#include <clf_perception_vision/ExtendedPersonStamped.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/image_encodings.h>

// FILTER
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>

// CV
// #include <opencv2/imgproc/imgproc.hpp>

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
        string depth_topic;
        string depth_info;

        virtual void onInit() {
            // Get the NodeHandle
            private_nh = getPrivateNodeHandle();
            if (private_nh.getParam("depth_topic", depth_topic))
            {
                ROS_INFO(">>> Input depth topic: %s", depth_topic.c_str());
            } else {
                ROS_ERROR("Failed to get depth topic");
            }
            if (private_nh.getParam("depth_info", depth_info))
            {
                ROS_INFO(">>> Input depth info topic: %s", depth_info.c_str());
            } else {
                ROS_ERROR("Failed to get depth info topic");
            }
            message_filters::Subscriber<Image> image_sub(private_nh, depth_topic.c_str(), 1);
            message_filters::Subscriber<CameraInfo> info_sub(private_nh, depth_info.c_str(), 1);
            TimeSynchronizer<Image, CameraInfo> sync(image_sub, info_sub, 10);
            sync.registerCallback(boost::bind(&DepthLookup::depth_callback, this ,_1, _2));
        }

        void depth_callback(const ImageConstPtr& image, const CameraInfoConstPtr& cam_info) {
            cout << "Callback depth" << endl;
        }

//        void setDepthData(const std::string &frameId, const ros::Time &stamp, const cv::Mat &depth, float depthConstant) {
//	        frameId_ = frameId;
//	        stamp_ = stamp;
//	        depth_ = depth;
//	        depthConstant_ = depthConstant;
//        }

        // const clf_perception_vision::ExtenedPeople::ConstPtr &person
        // This function is basically the one from here: https://github.com/introlab/find-object/blob/master/src/ros/FindObjectROS.cpp
//        cv::Vec3f get_depth(const cv::Mat & depthImage, int x, int y, float cx, float cy, float fx, float fy) {
//
//            if(!(x >=0 && x<depthImage.cols && y >=0 && y<depthImage.rows))
//            {
//                ROS_ERROR("Point must be inside the image (x=%d, y=%d), image size=(%d,%d)", x, y, depthImage.cols, depthImage.rows);
//                return cv::Vec3f( std::numeric_limits<float>::quiet_NaN (), std::numeric_limits<float>::quiet_NaN (), std::numeric_limits<float>::quiet_NaN ());
//            }
//
//            cv::Vec3f pt;
//
//            // Use correct principal point from calibration
//            float center_x = cx; //cameraInfo.K.at(2)
//            float center_y = cy; //cameraInfo.K.at(5)
//
//            bool isInMM = depthImage.type() == CV_16UC1; // is in mm?
//
//            // Combine unit conversion (if necessary) with scaling by focal length for computing (X,Y)
//            float unit_scaling = isInMM?0.001f:1.0f;
//            float constant_x = unit_scaling / fx; //cameraInfo.K.at(0)
//            float constant_y = unit_scaling / fy; //cameraInfo.K.at(4)
//            float bad_point = std::numeric_limits<float>::quiet_NaN ();
//
//            float depth;
//            bool isValid;
//            if(isInMM)
//            {
//                depth = (float)depthImage.at<uint16_t>(y,x);
//                isValid = depth != 0.0f;
//            }
//            else
//            {
//                depth = depthImage.at<float>(y,x);
//                isValid = std::isfinite(depth);
//            }
//
//            // Check for invalid measurements
//            if (!isValid)
//            {
//                pt.val[0] = pt.val[1] = pt.val[2] = bad_point;
//            }
//            else
//            {
//                // Fill in XYZ
//                pt.val[0] = (float(x) - center_x) * depth * constant_x;
//                pt.val[1] = (float(y) - center_y) * depth * constant_y;
//                pt.val[2] = depth*unit_scaling;
//            }
//            return pt;
//        }

    };

    PLUGINLIB_DECLARE_CLASS(clf_perception_vision, DepthLookup, clf_perception_vision::DepthLookup, nodelet::Nodelet
    );
}
