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
	    ROS_DEBUG("Image is in MM");
		depth = (float)depthImage.at<uint16_t>(y,x);
		isValid = depth != 0.0f;
	}
	else
	{
		ROS_DEBUG("Image is in M");
		depth = depthImage.at<float>(y,x);
		isValid = std::isfinite(depth);
	}

	// Check for invalid measurements
	if (!isValid)
	{
	    ROS_WARN("Image is invalid");
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
}

void syncCallback(const ImageConstPtr& depthMsg, const CameraInfoConstPtr& cameraInfoMsg, const ExtenedPeopleConstPtr& peopleMsg)
{
    tf::TransformBroadcaster tfBroadcaster_;
    std::vector<tf::StampedTransform> transforms;
    cv_bridge::CvImageConstPtr ptrDepth = cv_bridge::toCvShare(depthMsg);
    float depthConstant = 1.0f/cameraInfoMsg->K[4];
    setDepthData(depthMsg->header.frame_id, depthMsg->header.stamp, ptrDepth->image, depthConstant);
    int bbox_xmin, bbox_xmax, bbox_ymin, bbox_ymax;
    for (int i=0; i<peopleMsg->persons.size(); i++) {
        bbox_xmin = peopleMsg->persons[i].bbox_xmin;
        bbox_xmax = peopleMsg->persons[i].bbox_xmax;
        bbox_ymin = peopleMsg->persons[i].bbox_ymin;
        bbox_ymax = peopleMsg->persons[i].bbox_ymax;

        float objectWidth = bbox_xmax/2 - bbox_xmin/2;
        float objectHeight = bbox_ymax/2 - bbox_ymin/2;
        float center_x = (bbox_xmin/2+bbox_xmax/2)/2;
        float center_y = (bbox_ymin/2+bbox_ymax/2)/2;
        float xAxis_x = 3*objectWidth/4;
        float xAxis_y = objectHeight/2;
        float yAxis_x = objectWidth/2;
        float yAxis_y = 3*objectHeight/4;

        cv::Vec3f center3D = getDepth(depth_,
					center_x+0.5f, center_y+0.5f,
					float(depth_.cols/2)-0.5f, float(depth_.rows/2)-0.5f,
					1.0f/depthConstant_, 1.0f/depthConstant_);

        cv::Vec3f axisEndX = getDepth(depth_,
                xAxis_x+0.5f, xAxis_y+0.5f,
                float(depth_.cols/2)-0.5f, float(depth_.rows/2)-0.5f,
                1.0f/depthConstant_, 1.0f/depthConstant_);

        cv::Vec3f axisEndY = getDepth(depth_,
                yAxis_x+0.5f, yAxis_y+0.5f,
                float(depth_.cols/2)-0.5f, float(depth_.rows/2)-0.5f,
                1.0f/depthConstant_, 1.0f/depthConstant_);

        // cout << "Center3D" << center3D << endl;
        // cout << "axisEndX" << axisEndX << endl;
        // cout << "axisEndY" << axisEndY << endl;

        if(std::isfinite(center3D.val[0]) && std::isfinite(center3D.val[1]) && std::isfinite(center3D.val[2]) &&
				std::isfinite(axisEndX.val[0]) && std::isfinite(axisEndX.val[1]) && std::isfinite(axisEndX.val[2]) &&
				std::isfinite(axisEndY.val[0]) && std::isfinite(axisEndY.val[1]) && std::isfinite(axisEndY.val[2])) {
            tf::StampedTransform transform;
            transform.setIdentity();
            transform.child_frame_id_ = "Person";
            transform.frame_id_ = frameId_;
            transform.stamp_ = stamp_;
            transform.setOrigin(tf::Vector3(center3D.val[0], center3D.val[1], center3D.val[2]));

            //set rotation (y inverted)
            tf::Vector3 xAxis(axisEndX.val[0] - center3D.val[0], axisEndX.val[1] - center3D.val[1], axisEndX.val[2] - center3D.val[2]);
            xAxis.normalize();
            tf::Vector3 yAxis(axisEndY.val[0] - center3D.val[0], axisEndY.val[1] - center3D.val[1], axisEndY.val[2] - center3D.val[2]);
            yAxis.normalize();
            tf::Vector3 zAxis = xAxis*yAxis;
            tf::Matrix3x3 rotationMatrix(
                        xAxis.x(), yAxis.x() ,zAxis.x(),
                        xAxis.y(), yAxis.y(), zAxis.y(),
                        xAxis.z(), yAxis.z(), zAxis.z());
            tf::Quaternion q;
            rotationMatrix.getRotation(q);
            // set x axis going front of the object, with z up and z left
            q *= tf::createQuaternionFromRPY(CV_PI/2.0, CV_PI/2.0, 0);
            transform.setRotation(q.normalized());
            transforms.push_back(transform);
			} else {
				ROS_WARN("Object %d detected, center 2D at (%f,%f), but invalid depth, cannot set frame \"%s\"! "
						 "(maybe object is too near of the camera or bad depth image)\n",
						0,
						center_x, center_y,
						"Head");
			}

    }

    if(transforms.size()) {
	   tfBroadcaster_.sendTransform(transforms);
    }
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

    Subscriber<Image> image_sub(nh, depth_topic, 1);
    Subscriber<CameraInfo> info_sub(nh, depth_info, 1);
    Subscriber<ExtenedPeople> people_sub(nh, in_topic, 1);

    typedef sync_policies::ApproximateTime<Image, CameraInfo, ExtenedPeople> MySyncPolicy;

    Synchronizer<MySyncPolicy> sync(MySyncPolicy(5), image_sub, info_sub, people_sub);
    sync.registerCallback(boost::bind(&syncCallback, _1, _2, _3));
    people_pub = nh.advertise<ExtenedPeople>(out_topic, 1);

    ros::spin();
    return 0;
}