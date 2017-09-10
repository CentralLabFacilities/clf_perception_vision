#include "clf_perception_depth_lookup_objects.hpp"

using namespace cv;
using namespace std;
using namespace sensor_msgs;
using namespace geometry_msgs;
using namespace message_filters;
using namespace clf_perception_vision;


// This piece of code luckily already existed in https://github.com/introlab/find-object/blob/master/src/ros/FindObjectROS.cpp (THANKS!)
Vec3f getDepth(const Mat & depthImage, int x, int y, float cx, float cy, float fx, float fy) {
	if(!(x >=0 && x<depthImage.cols && y >=0 && y<depthImage.rows))
	{
		ROS_ERROR(">>> Point must be inside the image (x=%d, y=%d), image size=(%d,%d)", x, y, depthImage.cols, depthImage.rows);
		return Vec3f(
				numeric_limits<float>::quiet_NaN (),
				numeric_limits<float>::quiet_NaN (),
				numeric_limits<float>::quiet_NaN ());
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
	float bad_point = numeric_limits<float>::quiet_NaN ();

	float depth;
	bool isValid;

	if(isInMM)
	{
	    // ROS_DEBUG(">>> Image is in Millimeters");
		depth = (float)depthImage.at<uint16_t>(y,x);
		// ROS_DEBUG("%f", depth);
		isValid = depth != 0.0f;
	} else {
		// ROS_DEBUG(">>> Image is in Meters");
		depth = depthImage.at<float>(y,x);
		isValid = isfinite(depth);
	}

	// Check for invalid measurements
	if (!isValid) {
	    ROS_DEBUG(">>> WARN Image is invalid, whoopsie.");
		pt.val[0] = pt.val[1] = pt.val[2] = bad_point;
	} else {
		// Fill in XYZ
		pt.val[0] = (float(x) - center_x) * depth * constant_x;
		pt.val[1] = (float(y) - center_y) * depth * constant_y;
		pt.val[2] = depth*unit_scaling;
	}

	return pt;
}

void setDepthData(const string &frameId, const ros::Time &stamp, const Mat &depth, float depthConstant) {
    frameId_ = frameId;
    stamp_ = stamp;
    depth_ = depth;
    depthConstant_ = depthConstant;
}

void syncCallback(const ImageConstPtr& depthMsg,
                  const CameraInfoConstPtr& cameraInfoMsg,
                  const CameraInfoConstPtr& cameraInfoMsgRgb,
                  const ExtendedObjectsConstPtr& objectsMsg) {
    // Copy Message in order to manipulate it later and
    // sent updated version.
    ExtendedObjects objects_cpy;
    objects_cpy = *objectsMsg;
    vector<tf::StampedTransform> transforms;
    im_mutex.lock();

    cv_bridge::CvImageConstPtr ptrDepth;
    if (depthMsg->encoding == "16UC1") {
       ptrDepth = cv_bridge::toCvShare(depthMsg, sensor_msgs::image_encodings::TYPE_16UC1);
    } else if (depthMsg->encoding == "32FC1") {
       ptrDepth = cv_bridge::toCvShare(depthMsg, sensor_msgs::image_encodings::TYPE_32FC1);
    } else {
      ROS_ERROR(">>> Incompatible image encoding %s", depthMsg->encoding.c_str());
	  im_mutex.unlock();
	  return;
    }

    float depthConstant = 1.0f/cameraInfoMsg->K[4];
    setDepthData(depthMsg->header.frame_id, depthMsg->header.stamp, ptrDepth->image, depthConstant);
    int bbox_xmin, bbox_xmax, bbox_ymin, bbox_ymax;
    // If depth image and color image have different resolutions,
    // derive a factor to scale the bounding boxes
    float scale_factor = (cameraInfoMsgRgb->width/cameraInfoMsg->width);
    // ROS_INFO(">>> Scale ratio RGB Image to DEPTH image is: %f ", scale_factor);

    for (int i=0; i<objectsMsg->objects.size(); i++) {
        bbox_xmin = objectsMsg->objects[i].bbox_xmin;
        bbox_xmax = objectsMsg->objects[i].bbox_xmax;
        bbox_ymin = objectsMsg->objects[i].bbox_ymin;
        bbox_ymax = objectsMsg->objects[i].bbox_ymax;

        float objectWidth = bbox_xmax/scale_factor - bbox_xmin/scale_factor;
        float objectHeight = bbox_ymax/scale_factor - bbox_ymin/scale_factor;
        float center_x = (bbox_xmin/scale_factor+bbox_xmax/scale_factor)/2;
        float center_y = (bbox_ymin/scale_factor+bbox_ymax/scale_factor/shift_center_y)/2;

        // float xAxis_x = 3*objectWidth/4;
        // float xAxis_y = objectHeight/2;
        // float yAxis_x = objectWidth/2;
        // float yAxis_y = 3*objectHeight/4;

        cv::Vec3f center3D = getDepth(depth_,
					center_x+0.5f, center_y+0.5f,
					float(depth_.cols/2)-0.5f, float(depth_.rows/2)-0.5f,
					1.0f/depthConstant_, 1.0f/depthConstant_);

        // cv::Vec3f axisEndX = getDepth(depth_,
        //         xAxis_x+0.5f, xAxis_y+0.5f,
        //         float(depth_.cols/2)-0.5f, float(depth_.rows/2)-0.5f,
        //         1.0f/depthConstant_, 1.0f/depthConstant_);

        // cv::Vec3f axisEndY = getDepth(depth_,
        //         yAxis_x+0.5f, yAxis_y+0.5f,
        //         float(depth_.cols/2)-0.5f, float(depth_.rows/2)-0.5f,
        //         1.0f/depthConstant_, 1.0f/depthConstant_);

        string id = objectsMsg->objects[i].category + "_" + to_string(i);

        if(isfinite(center3D.val[0]) && isfinite(center3D.val[1]) && isfinite(center3D.val[2])) {

            tf::StampedTransform transform;
            transform.setIdentity();
            transform.child_frame_id_ = id;
            transform.frame_id_ = frameId_;
            transform.stamp_ = stamp_;
            transform.setOrigin(tf::Vector3(center3D.val[0], center3D.val[1], center3D.val[2]));

            // set rotation (y inverted)
            // tf::Vector3 xAxis(axisEndX.val[0] - center3D.val[0], axisEndX.val[1] - center3D.val[1], axisEndX.val[2] - center3D.val[2]);
            // xAxis.normalize();
            // tf::Vector3 yAxis(axisEndY.val[0] - center3D.val[0], axisEndY.val[1] - center3D.val[1], axisEndY.val[2] - center3D.val[2]);
            // yAxis.normalize();
            // tf::Vector3 zAxis = xAxis*yAxis;
            // tf::Matrix3x3 rotationMatrix(
            //            xAxis.x(), yAxis.x(), zAxis.x(),
            //            xAxis.y(), yAxis.y(), zAxis.y(),
            //            xAxis.z(), yAxis.z(), zAxis.z());
            // tf::Quaternion q;
            // rotationMatrix.getRotation(q);
            // set x axis going front of the object, with z up and z left
            // q *= tf::createQuaternionFromRPY(CV_PI/2.0, CV_PI/2.0, 0);
            // transform.setRotation(q.normalized());

            transforms.push_back(transform);

            PoseStamped pose_stamped;
            pose_stamped.header = objects_cpy.header;
            pose_stamped.header.frame_id = frameId_;

            pose_stamped.pose.position.x = center3D.val[0];
            pose_stamped.pose.position.y = center3D.val[1];
            pose_stamped.pose.position.z = center3D.val[2];

            pose_stamped.pose.orientation.x = 0.0; //q.normalized().x();
            pose_stamped.pose.orientation.y = 0.0; //q.normalized().y();
            pose_stamped.pose.orientation.z = 0.0; //q.normalized().z();
            pose_stamped.pose.orientation.w = 1.0; //q.normalized().w();

            objects_cpy.objects[i].pose = pose_stamped;
            objects_cpy.objects[i].transformid = id;

            ROS_DEBUG(">>> person_%d detected, center 2D at (%f,%f) setting frame \"%s\" \n", i, center_x, center_y, id.c_str());
		} else {
			ROS_DEBUG(">>> WARN person_%d detected, center 2D at (%f,%f), but invalid depth, cannot set frame \"%s\"!\n", i, center_x, center_y, id.c_str());
		}
    }

    im_mutex.unlock();

    if(transforms.size()) {
	   tfBroadcaster_->sendTransform(transforms);
	   objects_pub.publish(objects_cpy);
    }

}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "clf_perception_depth_lookup", ros::init_options::AnonymousName);
    ros::NodeHandle nh("~");

    if (nh.getParam("depthlookup_image_topic", depth_topic))
    {
        ROS_INFO(">>> Input depth image topic: %s", depth_topic.c_str());
    } else {
        ROS_ERROR("!Failed to get depth image topic parameter!");
        exit(EXIT_FAILURE);
    }

    if (nh.getParam("depthlookup_depth_info_topic", depth_info))
    {
        ROS_INFO(">>> Input depth camera info topic: %s", depth_info.c_str());
    } else {
        ROS_ERROR("!Failed to get depth camera info topic parameter!");
        exit(EXIT_FAILURE);
    }

    if (nh.getParam("depthlookup_rgb_info_topic", rgb_info))
    {
        ROS_INFO(">>> Input rgb camera info topic: %s", rgb_info.c_str());
    } else {
        ROS_ERROR("!Failed to get rgb camera info topic parameter!");
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

    if (nh.getParam("depthlookup_shift_center_y", shift_center_y))
    {
        ROS_INFO(">>> Shift center_y: %f", shift_center_y);
    } else {
        ROS_ERROR("!Failed to get output topic parameter!");
        exit(EXIT_FAILURE);
    }

    tfBroadcaster_ = new tf::TransformBroadcaster();

    Subscriber<Image> image_sub(nh, depth_topic, 1);
    Subscriber<CameraInfo> info_depth_sub(nh, depth_info, 1);
    Subscriber<CameraInfo> info_rgb_sub(nh, rgb_info, 1);
    Subscriber<ExtendedObjects> objects_sub(nh, in_topic, 1);

    typedef sync_policies::ApproximateTime<Image, CameraInfo, CameraInfo, ExtendedObjects> sync_pol;
    Synchronizer<sync_pol> sync(sync_pol(5), image_sub, info_depth_sub, info_rgb_sub, objects_sub);
    sync.registerCallback(boost::bind(&syncCallback, _1, _2, _3, _4));

    objects_pub = nh.advertise<ExtendedObjects>(out_topic, 1);

    // namedWindow(":: CLF DEPTH LOOKUP OBJECTS ::", cv::WINDOW_AUTOSIZE);

    ros::spin();

    return 0;
}