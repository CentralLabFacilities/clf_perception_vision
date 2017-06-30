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

Size minSize(60,60);
Size maxSize(200,200);

static void help()
{
    cout << "Usage: ./clf_faces_ros \n\t--cascade <cascade_file>\n\t--topic <ros_topic>)\n"
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


static void displayState(Mat &canvas, double scaleFactor ,double fps)
{
    Scalar fontColorWhite = CV_RGB(255,255,255);
    Scalar fontColorNV  = CV_RGB(135,206,250);

    ostringstream ss;

    ss << "FPS = " << setprecision(1) << fixed << fps;
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
    toggle_sub = nh_.subscribe("/clf_faces/people/subscribe", 1, toggle_callback);

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

    string cascadeName;
    string topic;

    for (int i = 1; i < argc; ++i)
    {
        if (string(argv[i]) == "--cascade")
        {
            cascadeName = argv[++i];
            cout << ">>> Cascadename " << cascadeName << endl;
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

    ROSGrabber ros_grabber(topic);

    namedWindow(":: CLF GPU Face Detect [ROS] Press q to Exit ::", 1);

    Mat frame, frameDisp;
    GpuMat frame_cuda, frame_cuda_grey, facesBuf_cuda;

    double scaleFactor = 1.2;
    bool findLargestObject = true;
    std::string duration;

    while(run) {

        boost::posix_time::ptime init = boost::posix_time::microsec_clock::local_time();

        ros::spinOnce();

        if(toggle) {

            ros_grabber.getImage(&frame);

            if (frame.empty())
            {
                continue;
            }

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

            std::vector<Rect> faces;
            cascade_cuda->convert(facesBuf_cuda, faces);

            if(draw) {
                for(int i = 0; i < faces.size(); ++i)
                    cv::rectangle(frame, faces[i], cv::Scalar(255), 4);
            }

            std_msgs::Header h;
            h.stamp = ros_grabber.getTimestamp();
            h.frame_id = "0";

            // ROS MSGS
            people_msgs::People people_msg;
            people_msgs::Person person_msg;
            people_msg.header = h;

            double face_size = 0.0;

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

            people_pub.publish(people_msg);

            tm.stop();
            double detectionTime = tm.getTimeMilli();
            double fps = 1000 / detectionTime;

            if(draw) {
                displayState(frameDisp, scaleFactor, fps);
                imshow(":: CLF GPU Face Detect [ROS] Press q to Exit ::", frameDisp);
            }

        }

        char key = (char)waitKey(1);

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

        boost::posix_time::ptime c = boost::posix_time::microsec_clock::local_time();
        boost::posix_time::time_duration cdiff = c - init;
        duration = std::to_string(cdiff.total_milliseconds());
    }

    return 0;
}
