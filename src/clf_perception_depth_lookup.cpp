#include "clf_perception_depth_lookup.hpp"

using namespace cv;
using namespace std;
using namespace sensor_msgs;
using namespace message_filters;
using namespace clf_perception_vision;

// This piece of code luckily already existed in https://github.com/introlab/find-object/blob/master/src/ros/FindObjectROS.cpp (THANKS!)
cv::Vec3f getDepth(const cv::Mat & depthImage, int x, int y, float cx, float cy, float fx, float fy) {
	if(!(x >=0 && x<depthImage.cols && y >=0 && y<depthImage.rows))
	{
		ROS_ERROR("Point must be inside the image (x=%d, y=%d), image size=(%d,%d)", x, y, depthImage.cols, depthImage.rows);
		return cv::Vec3f(
				std::numeric_limits<float>::quiet_NaN (),
				std::numeric_limits<float>::quiet_NaN (),
				std::numeric_limits<float>::quiet_NaN ());
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
	float bad_point = std::numeric_limits<float>::quiet_NaN ();

	float depth;
	bool isValid;

	if(isInMM)
	{
		depth = (float)depthImage.at<uint16_t>(y,x);
		isValid = depth != 0.0f;
	}
	else
	{
		depth = depthImage.at<float>(y,x);
		isValid = std::isfinite(depth);
	}

	// Check for invalid measurements
	if (!isValid)
	{
		pt.val[0] = pt.val[1] = pt.val[2] = bad_point;
	}
	else
	{
		// Fill in XYZ
		pt.val[0] = (float(x) - center_x) * depth * constant_x;
		pt.val[1] = (float(y) - center_y) * depth * constant_y;
		pt.val[2] = depth*unit_scaling;
	}
	return pt;
}

void setDepthData(const std::string &frameId, const ros::Time &stamp, const cv::Mat &depth, float depthConstant) {
    frameId_ = frameId;
    stamp_ = stamp;
    depth_ = depth;
    depthConstant_ = depthConstant;
    cv::Vec3f point;
    point = getDepth(depth_,0.5f, 0.5f, float(depth_.cols/2)-0.5f, float(depth_.rows/2)-0.5f, 1.0f/depthConstant_, 1.0f/depthConstant_);
    // cout << point << endl;
}

void syncCallback(const ImageConstPtr& depthMsg, const CameraInfoConstPtr& cameraInfoMsg, const ExtenedPeopleConstPtr& peopleMsg)
{
    cv_bridge::CvImageConstPtr ptrDepth = cv_bridge::toCvShare(depthMsg);
    float depthConstant = 1.0f/cameraInfoMsg->K[4];
    setDepthData(depthMsg->header.frame_id, depthMsg->header.stamp, ptrDepth->image, depthConstant);
    cout << peopleMsg << endl;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "clf_perception_depth_lookup", ros::init_options::AnonymousName);
    ros::Publisher people_pub;
    ros::NodeHandle nh("~");

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
        ROS_INFO(">>> Output Topic: %s", out_topic.c_str());
    } else {
        ROS_ERROR("!Failed to get output topic parameter!");
        exit(EXIT_FAILURE);
    }

    if (nh.getParam("depthlookup_in_topic", in_topic))
    {
        ROS_INFO(">>> Input Topic: %s", in_topic.c_str());
    } else {
        ROS_ERROR("!Failed to get input topic parameter!");
        exit(EXIT_FAILURE);
    }

    Subscriber<Image> image_sub(nh, depth_topic, 5);
    Subscriber<CameraInfo> info_sub(nh, depth_info, 5);
    Subscriber<ExtenedPeople> people_sub(nh, in_topic, 5);

    TimeSynchronizer<Image, CameraInfo, ExtenedPeople> sync(image_sub, info_sub, people_sub, 5);
    sync.registerCallback(boost::bind(syncCallback, _1, _2, _3));

    people_pub = nh.advertise<ExtenedPeople>(out_topic, 5);

    ros::spin();
    return 0;
}