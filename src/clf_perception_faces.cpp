/*

Author: Florian Lier [flier AT techfak.uni-bielefeld DOT de]

By downloading, copying, installing or using the software you agree to this license.
If you do not agree to this license, do not download, install, copy or use the software.

                          License Agreement
               For Open Source Computer Vision Library
                       (3-clause BSD License)

Copyright (C) 2000-2016, Intel Corporation, all rights reserved.
Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
Copyright (C) 2009-2016, NVIDIA Corporation, all rights reserved.
Copyright (C) 2010-2013, Advanced Micro Devices, Inc., all rights reserved.
Copyright (C) 2015-2016, OpenCV Foundation, all rights reserved.
Copyright (C) 2015-2016, Itseez Inc., all rights reserved.
Third party copyrights are property of their respective owners.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

  * Neither the names of the copyright holders nor the names of the contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.

This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are disclaimed.
In no event shall copyright holders or contributors be liable for any direct,
indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.

*/

// SELF
#include "ros_grabber.hpp"

// STD
#include <iostream>
#include <string>
#include <iostream>
#include <iomanip>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// CUDA
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/xfeatures2d/cuda.hpp>
#include <opencv2/cudaobjdetect.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>


#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>

// ROS
#include <ros/ros.h>
#include <std_msgs/Bool.h>
#include <sensor_msgs/Image.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/PointStamped.h>
#include <clf_perception_vision_msgs/ExtendedPeople.h>
#include <clf_perception_vision_msgs/ExtendedPersonStamped.h>
#include <clf_perception_vision_msgs/ImageToPose.h>
#include <bayes_people_tracker_msgs/PeopleTrackerImage.h>

// BOOST
#include "boost/date_time/posix_time/posix_time.hpp"

using namespace std;
using namespace cv;
using namespace cuda;
using namespace sensor_msgs;
using namespace geometry_msgs;
using namespace clf_perception_vision_msgs;

bool toggle = true;

std::string out_topic_face;
std::string in_topic_bounding_boxes;
std::string in_topic_image, cascade_profile_file, cascade_frontal_file, depth_info, imageToPoseClient_topic;

Mat frame;
bayes_people_tracker_msgs::PeopleTrackerImage peopleTrackerImages;
std::mutex personMutex;

unsigned int average_frames = 0;
unsigned int last_computed_frame = -1;
unsigned int frame_count = 0;
unsigned int min_n = 2;

double scaleFactor = 1.4;
double time_spend = 0;

bool findLargestObject = true;
bool visualize = false;

Size minSize(20,20);
Size maxSize(100,100);

const int fontFace = FONT_HERSHEY_PLAIN;
const double fontScale = 1;

ros::Publisher face_pub;
ros::ServiceClient imageToPoseClient;

static void help()
{
    cout << "Usage: <proc> \n\t config.yaml\n" << endl;
}

static void matPrint(Mat &img, int lineOffsY, Scalar fontColor, const string &ss)
{
    int fontFace = FONT_HERSHEY_DUPLEX;
    double fontScale = 0.45;
    int fontThickness = 0.2;
    Size fontSize = getTextSize("T[]", fontFace, fontScale, fontThickness, 0);

    cv::Point org;
    org.x = 1;
    org.y = 3 * fontSize.height * (lineOffsY + 1) / 2;
    putText(img, ss, org, fontFace, fontScale, CV_RGB(0,0,0), 5*fontThickness/2, 16);
    putText(img, ss, org, fontFace, fontScale, fontColor, fontThickness, 16);
}

void toggle_callback(const std_msgs::Bool& _toggle) {
    toggle = _toggle.data;
    cout << ">>> I am currently computing? --> " << toggle << endl;
}

static void displayState(Mat &canvas, double scaleFactor)
{
    Scalar fontColorWhite = CV_RGB(255,255,255);
    Scalar fontColorNV  = CV_RGB(135,206,250);

    ostringstream ss;

    ss << "FPS = " << setprecision(1) << fixed << average_frames;
    matPrint(canvas, 0, fontColorWhite, ss.str());
    ss.str("");
    ss << "[" << canvas.cols << "x" << canvas.rows << "] | " << "ScaleFactor " << scaleFactor;

    matPrint(canvas, 1, fontColorWhite, ss.str());
}

void getFaceCb(const bayes_people_tracker_msgs::PeopleTrackerImage &msg) {
    personMutex.lock();

    peopleTrackerImages = msg;
    cv_bridge::CvImagePtr cvBridge;

    Ptr<cuda::CascadeClassifier> cascade_cuda = cuda::CascadeClassifier::create(cascade_frontal_file);
    Ptr<cuda::CascadeClassifier> cascade_cuda_profile = cuda::CascadeClassifier::create(cascade_profile_file);

    namedWindow("CLF PERCEPTION || Face", 1);

    Mat frame, frame_display, to_extract;
    GpuMat frame_cuda, frame_cuda_grey, facesBuf_cuda, facesBuf_cuda_profile;

    for(int i = 0; i< peopleTrackerImages.trackedPeopleImg.size(); i++){

        try {
            cvBridge = cv_bridge::toCvCopy(peopleTrackerImages.trackedPeopleImg.at(i).image, sensor_msgs::image_encodings::BGR8);
        }
        catch (cv_bridge::Exception &e) {
            ROS_ERROR("cv_bridge failed to convert sensor msg: %s", e.what());
            personMutex.unlock();
        }
        frame = cvBridge->image;
        
        time_t start, end;
        time(&start);

        if (frame.rows*frame.cols > 0) {
            if(toggle) {

                frame_display = frame.clone();
                frame_cuda.upload(frame);
                cuda::cvtColor(frame_cuda, frame_cuda_grey, COLOR_BGR2GRAY);

                TickMeter tm;
                tm.start();

                cascade_cuda->setMinNeighbors(min_n);
                cascade_cuda->setScaleFactor(scaleFactor);
                cascade_cuda->setFindLargestObject(true);
                cascade_cuda->setMinObjectSize(minSize);
                cascade_cuda->setMaxObjectSize(maxSize);
                cascade_cuda->detectMultiScale(frame_cuda_grey, facesBuf_cuda);

                cascade_cuda_profile->setMinNeighbors(min_n);
                cascade_cuda_profile->setScaleFactor(scaleFactor);
                cascade_cuda_profile->setFindLargestObject(true);
                cascade_cuda_profile->setMinObjectSize(minSize);
                cascade_cuda_profile->setMaxObjectSize(maxSize);
                cascade_cuda_profile->detectMultiScale(frame_cuda_grey, facesBuf_cuda_profile);

                std::vector<Rect> faces;
                std::vector<Rect> faces_profile;

                cascade_cuda->convert(facesBuf_cuda, faces);

                if(visualize) {
                    if (faces.size() > 0) {
                        for(int i = 0; i < faces.size(); ++i) {
                            rectangle(frame_display, faces[i], Scalar(113,179,60), 3);
                        }
                    } else {
                    cascade_cuda_profile->convert(facesBuf_cuda_profile, faces_profile);
                    if (faces_profile.size() > 0) {
                        for(int i = 0; i < faces_profile.size(); ++i) {
                            rectangle(frame_display, faces_profile[i], Scalar(102,255,255), 3);
                        }
                    }
                    }
                }

                frame_count++;
                tm.stop();
                double detectionTime = tm.getTimeMilli();
                double fps = 1000 / detectionTime;

                if(visualize) {
                    imshow("CLF PERCEPTION || Face", frame_display);
                    cv::waitKey(3);
                }

                if (time_spend >= 1 ) {
                    average_frames = frame_count;
                    time(&start);
                    frame_count = 0;
                }

                time(&end);
                time_spend = difftime(end, start);

                if (faces.size() > 0) {
                    if (faces.size() > 1) {
                        ROS_WARN(">>> Detected more than one face for this person. Aborting...");
                        return;
                    }

                    //const ImageConstPtr& depthMsg = peopleTrackerImages.trackedPeopleImg.at(i).image;
                    Image depthMsg = peopleTrackerImages.trackedPeopleImg.at(i).image;
                    cv_bridge::CvImageConstPtr ptrDepth;

                    try {
                        if (depthMsg.encoding == "16UC1") {
                        ptrDepth = cv_bridge::toCvCopy(depthMsg, sensor_msgs::image_encodings::TYPE_16UC1);
                        } else if (depthMsg.encoding == "32FC1") {
                        ptrDepth = cv_bridge::toCvCopy(depthMsg, sensor_msgs::image_encodings::TYPE_32FC1);
                        } else {
                        ROS_ERROR(">>> Unknown image encoding %s", depthMsg.encoding.c_str());
                        personMutex.unlock();
                        return;
                        }
                    } catch (cv_bridge::Exception& e) {
                        ROS_ERROR(">>> CV_BRIDGE exception: %s", e.what());
                    return;
                    }

                    Rect face = faces[0];
                    cv::Mat croppedImage_depth = ptrDepth->image(face);

                    cv_bridge::CvImage image_depth_out_msg;
                    image_depth_out_msg.header   = peopleTrackerImages.header;
                    if (depthMsg.encoding == "16UC1") {
                        image_depth_out_msg.encoding = sensor_msgs::image_encodings::TYPE_16UC1;
                    } else if (depthMsg.encoding == "32FC1") {
                        image_depth_out_msg.encoding = sensor_msgs::image_encodings::TYPE_32FC1;
                    }
                    
                    image_depth_out_msg.image = croppedImage_depth;

                    ROS_INFO("Sending service call...");

                    clf_perception_vision_msgs::ImageToPose srv;
                    srv.request.image_depth = *image_depth_out_msg.toImageMsg();
                    if (imageToPoseClient.call(srv))
                    {
                        ROS_INFO("Got face pose!");
                        face_pub.publish(srv.response.pose_stamped);
                    }
                    else
                    {
                        ROS_ERROR("Failed to call image to pose service!");
                        personMutex.unlock();
                        return;
                    }
                }   

            }
        }
    }
    personMutex.unlock();

}

int main(int argc, char *argv[])
{

    ros::init(argc, argv, "clf_detect_faces");
    ros::NodeHandle private_node_handle("~");
    
    private_node_handle.param("imageToPoseClient_topic", imageToPoseClient_topic, std::string("/clf_perception_vision/get_pose_from_image"));
    private_node_handle.param("visualize", visualize, false);
    private_node_handle.param("bounding_box_topic", in_topic_bounding_boxes, std::string("/people_tracker/people/extended"));
    private_node_handle.param("publish_topic", out_topic_face, std::string("/clf_perception_vision/people/head_extended"));
    private_node_handle.param("cascade_frontal_file", cascade_frontal_file, std::string("/home/pepper/citk/systems/pepper-robocup-nightly//share/clf_perception_vision/data/haar/haarcascade_frontalface_default.xml"));
    private_node_handle.param("cascade_profile_file", cascade_profile_file, std::string("/home/pepper/citk/systems/pepper-robocup-nightly//share/clf_perception_vision/data/haar/haarcascade_profileface.xml"));
    
    ROS_INFO("imageToPoseClient_topic set to %s", imageToPoseClient_topic.c_str());
    ROS_INFO("bounding_box_topic set to %s", in_topic_bounding_boxes.c_str());
    ROS_INFO("publish_topic set to %s", out_topic_face.c_str());
    ROS_INFO("cascade_frontal_file set to %s", cascade_frontal_file.c_str());
    ROS_INFO("cascade_profile_file set to %s", cascade_profile_file.c_str());

    face_pub = private_node_handle.advertise<PoseStamped>(out_topic_face, 1);

    imageToPoseClient = private_node_handle.serviceClient<clf_perception_vision_msgs::ImageToPose>(imageToPoseClient_topic);

    ros::NodeHandle n;
    // subscriber to recieve extended person message
    ros::Subscriber extendedPeopleSub = private_node_handle.subscribe("people_tracker/people/extended", 1, getFaceCb);

    if (getCudaEnabledDeviceCount() == 0)
    {
        return cerr << ">>> No GPU found or OpenCV is compiled without GPU support" << endl, -1;
    }

    cout << ">>> Cuda Enabled Devices --> " << cuda::getCudaEnabledDeviceCount() << endl;
    cout << ">>> ";

    cuda::printShortCudaDeviceInfo(cuda::getDevice());

    ROS_INFO("Init done. Can start detecting faces.");
    ros::spin();
    
    return 0;
}
