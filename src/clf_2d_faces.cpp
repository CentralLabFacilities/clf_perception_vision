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

// ROS
#include <ros/ros.h>
#include <std_msgs/Bool.h>
#include <people_msgs/People.h>
#include <people_msgs/Person.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/PointStamped.h>

// BOOST
#include "boost/date_time/posix_time/posix_time.hpp"

using namespace std;
using namespace cv;
using namespace cv::cuda;

bool toggle = true;
bool draw = true;
bool run = true;

unsigned int average_frames = 0;
unsigned int last_computed_frame = -1;
unsigned int frame_count = 0;

double scaleFactor = 1.2;
double time_spend = 0;

bool findLargestObject = true;

Size minSize(40,40);
Size maxSize(200,200);

static void help()
{
    cout << "Usage: ./clf_faces_ros \n\t--cascade <cascade_file>\n\t--cascade-profile <cascade_file>\n\t--topic <ros_topic>)\n"
            "Using OpenCV version " << CV_VERSION << endl << endl;
}

static void matPrint(Mat &img, int lineOffsY, Scalar fontColor, const string &ss)
{
    int fontFace = FONT_HERSHEY_DUPLEX;
    double fontScale = 0.45;
    int fontThickness = 0.2;
    Size fontSize = cv::getTextSize("T[]", fontFace, fontScale, fontThickness, 0);

    Point org;
    org.x = 1;
    org.y = 3 * fontSize.height * (lineOffsY + 1) / 2;
    putText(img, ss, org, fontFace, fontScale, CV_RGB(0,0,0), 5*fontThickness/2, 16);
    putText(img, ss, org, fontFace, fontScale, fontColor, fontThickness, 16);
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

void toggle_callback(const std_msgs::Bool& _toggle) {
    toggle = _toggle.data;
    cout << ">>> I am currently computing? --> " << toggle << endl;
}

int main(int argc, char *argv[])
{

    ros::init(argc, argv, "clf_faces", ros::init_options::AnonymousName);

    ros::NodeHandle nh_;
    ros::Publisher people_pub;
    ros::Subscriber toggle_sub;

    people_pub = nh_.advertise<people_msgs::People>("clf_faces/people", 20);
    toggle_sub = nh_.subscribe("/clf_faces/people/compute", 1, toggle_callback);

    if (argc == 1)
    {
        help();
        return -1;
    }

    if (getCudaEnabledDeviceCount() == 0)
    {
        return cerr << ">>> No GPU found or the library is compiled without GPU support" << endl, -1;
    }

    cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());

    string cascadeName, cascadeNameProfile;
    string topic;

    for (int i = 1; i < argc; ++i)
    {
        if (string(argv[i]) == "--cascade")
        {
            cascadeName = argv[++i];
            cout << ">>> Cascadename " << cascadeName << endl;
        }
        else if (string(argv[i]) == "--cascade-profile")
        {
            cascadeNameProfile = argv[++i];
            cout << ">>> Cascadename Profile " << cascadeNameProfile << endl;
        }
        else if (string(argv[i]) == "--topic")
        {
            topic = argv[++i];
            cout << ">>> Input Topic " << topic << endl;
        }
        else if (string(argv[i]) == "--help")
        {
            help();
            return -1;
        }
        else
        {
            cout << ">>> Unknown key: " << argv[i] << endl;
            return -1;
        }
    }

    Ptr<cuda::CascadeClassifier> cascade_cuda = cuda::CascadeClassifier::create(cascadeName);
    Ptr<cuda::CascadeClassifier> cascade_cuda_profile = cuda::CascadeClassifier::create(cascadeNameProfile);

    ROSGrabber ros_grabber(topic);

    namedWindow(":: CLF GPU Face Detect [ROS] Press q to Exit ::", 1);

    Mat frame, frameDisp;
    GpuMat frame_cuda, frame_cuda_grey, facesBuf_cuda, facesBuf_cuda_profile;

    time_t start, end;
    time(&start);

    while(run) {

        ros::spinOnce();

        char key = (char)waitKey(1);

        if(toggle) {

            ros_grabber.getImage(&frame);

            if (frame.rows*frame.cols > 0) {

                int tmp_frame_nr = ros_grabber.getLastFrameNr();

                if(last_computed_frame != tmp_frame_nr) {

                    frameDisp = frame.clone();
                    frame_cuda.upload(frame);

                    cv::cuda::cvtColor(frame_cuda, frame_cuda_grey, cv::COLOR_BGR2GRAY);

                    TickMeter tm;
                    tm.start();

                    cascade_cuda->setMinNeighbors(4);
                    cascade_cuda->setScaleFactor(scaleFactor);
                    cascade_cuda->setFindLargestObject(true);
                    cascade_cuda->setMinObjectSize(minSize);
                    cascade_cuda->setMaxObjectSize(maxSize);
                    cascade_cuda->detectMultiScale(frame_cuda_grey, facesBuf_cuda);

                    cascade_cuda_profile->setMinNeighbors(4);
                    cascade_cuda_profile->setScaleFactor(scaleFactor);
                    cascade_cuda_profile->setFindLargestObject(true);
                    cascade_cuda_profile->setMinObjectSize(minSize);
                    cascade_cuda_profile->setMaxObjectSize(maxSize);
                    cascade_cuda_profile->detectMultiScale(frame_cuda_grey, facesBuf_cuda_profile);

                    std::vector<Rect> faces;
                    std::vector<Rect> faces_profile;

                    cascade_cuda->convert(facesBuf_cuda, faces);

                    if(draw) {
                        if (faces.size() > 0) {
                            for(int i = 0; i < faces.size(); ++i) {
                                cv::rectangle(frameDisp, faces[i], cv::Scalar(250,206,35), 4);
                            }
                        } else {
                          cascade_cuda_profile->convert(facesBuf_cuda_profile, faces_profile);
                          if (faces_profile.size() > 0) {
                              for(int i = 0; i < faces_profile.size(); ++i) {
                                cv::rectangle(frameDisp, faces_profile[i], cv::Scalar(226,43,138), 4);
                              }
                          }
                        }
                    }

                    std_msgs::Header h;
                    h.stamp = ros_grabber.getTimestamp();
                    h.frame_id = ros_grabber.frame_id;

                    // ROS MSGS
                    people_msgs::People people_msg;
                    people_msgs::Person person_msg;
                    people_msg.header = h;
                    
                    if (faces.size() > 0) {
                        for (int i = 0; i < faces.size(); ++i) {
                            person_msg.name = "unknown";
                            person_msg.reliability = 0.0;
                            geometry_msgs::Point p;
                            Point center = Point(faces[i].x + faces[i].width/2.0, faces[i].y + faces[i].height/2.0);
                            double mid_x = center.x;
                            double mid_y = center.y;
                            p.x = center.x;
                            p.y = center.y;
                            p.z = faces[i].size().area();
                            person_msg.position = p;
                            people_msg.people.push_back(person_msg);
                        }
                    } else if (faces_profile.size() > 0) {
                        for (int i = 0; i < faces_profile.size(); ++i) {
                            person_msg.name = "unknown";
                            person_msg.reliability = 0.0;
                            geometry_msgs::Point p;
                            Point center = Point(faces_profile[i].x + faces_profile[i].width/2.0, faces_profile[i].y + faces_profile[i].height/2.0);
                            double mid_x = center.x;
                            double mid_y = center.y;
                            p.x = center.x;
                            p.y = center.y;
                            p.z = faces_profile[i].size().area();
                            person_msg.position = p;
                            people_msg.people.push_back(person_msg);
                        }
                    }

                    people_pub.publish(people_msg);

                    frame_count++;

                    tm.stop();
                    double detectionTime = tm.getTimeMilli();
                    double fps = 1000 / detectionTime;

                    if(draw) {
                        displayState(frameDisp, scaleFactor);
                        imshow(":: CLF GPU Face Detect [ROS] Press q to Exit ::", frameDisp);
                    }

                    last_computed_frame = ros_grabber.getLastFrameNr();
                }
            }

            if (time_spend >= 1 ) {
                average_frames = frame_count;
                time(&start);
                frame_count = 0;
            }

            time(&end);
            time_spend = difftime(end, start);

        }

        switch (key)
        {
        case '+':
            scaleFactor *= 1.05;
            break;
        case '-':
            if (scaleFactor <= 1.01) {
                break;
            }
            scaleFactor /= 1.05;
            break;
        case 's':
        case 'S':
            draw = !draw;
            break;
        case 'q':
            run = !run;
            break;
        }

    }

    return 0;
}
