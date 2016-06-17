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

#if defined _MSC_VER && _MSC_VER >= 1400
#pragma warning(disable : 4100)
#endif

// STD
#include <iostream>
#include <iomanip>
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/gpu/gpu.hpp"

// ROS
#include <ros/ros.h>
#include <std_msgs/Bool.h>
#include "ros_grabber.hpp"
#include "people_msgs/People.h"
#include "people_msgs/Person.h"
#include "geometry_msgs/Point.h"
#include "geometry_msgs/PointStamped.h"

// BOOST
#include "boost/date_time/posix_time/posix_time.hpp"

using namespace std;
using namespace cv;
using namespace cv::gpu;

bool toggle = true;
bool draw = false;


static void help()
{
    cout << "Usage: ./clf_2d_faces_ros \n\t--cascade <cascade_file>\n\t--topic <ros_topic>)\n"
            "Using OpenCV version " << CV_VERSION << endl << endl;
}


template<class T>
void convertAndResize(const T& src, T& gray, T& resized, double scale)
{
    if (src.channels() == 3)
    {
        cvtColor( src, gray, CV_BGR2GRAY );
    }
    else
    {
        gray = src;
    }

    Size sz(cvRound(gray.cols * scale), cvRound(gray.rows * scale));

    if (scale != 1)
    {
        resize(gray, resized, sz);
    }
    else
    {
        resized = gray;
    }
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


static void displayState(Mat &canvas, bool bHelp, bool bGpu, bool bLargestFace, bool bFilter, double fps)
{
    Scalar fontColorWhite = CV_RGB(255,255,255);
    Scalar fontColorNV  = CV_RGB(135,206,250);

    ostringstream ss;

    ss << "FPS = " << setprecision(1) << fixed << fps;
    matPrint(canvas, 0, fontColorWhite, ss.str());
    ss.str("");
    ss << "[" << canvas.cols << "x" << canvas.rows << "] | " <<
        (bLargestFace ? "OneFace | " : "MultiFace | ") <<
        (bFilter ? "Filter:ON" : "Filter:OFF");

    matPrint(canvas, 1, fontColorWhite, ss.str());

    if (bHelp)
    {
        // matPrint(canvas, 3, fontColorNV, "M - switch OneFace / MultiFace");
        // matPrint(canvas, 4, fontColorNV, "F - toggle rectangles Filter");
        // matPrint(canvas, 5, fontColorNV, "H - toggle hotkeys help");
        // matPrint(canvas, 6, fontColorNV, "1/Q - increase/decrease scale");
    }
    else
    {
        // matPrint(canvas, 2, fontColorNV, "H - toggle hotkeys help");
    }
}

void toggle_callback(const std_msgs::Bool& _toggle) {
    toggle = _toggle.data;
    cout << ">>> I am currently computing? --> " << toggle << endl;
}

int main(int argc, char *argv[])
{

    ros::init(argc, argv, "clf_2d_faces", ros::init_options::AnonymousName);

    ros::NodeHandle nh_;
    ros::Publisher people_pub;
    ros::Subscriber toggle_sub;

    people_pub = nh_.advertise<people_msgs::People>("clf_2d_detect/people", 20);
    toggle_sub = nh_.subscribe("/clf_2d_detect/people/subscribe", 1, toggle_callback);

    if (argc == 1)
    {
        help();
        return -1;
    }

    if (getCudaEnabledDeviceCount() == 0)
    {
        return cerr << ">>> No GPU found or the library is compiled without GPU support" << endl, -1;
    }

    cv::gpu::printShortCudaDeviceInfo(cv::gpu::getDevice());

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

    CascadeClassifier_GPU cascade_gpu;

    if (!cascade_gpu.load(cascadeName))
    {
        return cerr << ">>> ERROR: Could not load GPU cascade classifier \"" << cascadeName << "\"" << endl, help(), -1;
    }

    ROSGrabber ros_grabber(topic);

    namedWindow(":: CLF GPU Face Detect [ROS] ::", 1);

    Mat frame, gray_cpu, resized_cpu, faces_downloaded, frameDisp, image;
    vector<Rect> facesBuf_cpu;

    GpuMat frame_gpu, gray_gpu, resized_gpu, facesBuf_gpu;

    bool useGPU = true;
    double scaleFactor = 1.0;
    bool findLargestObject = true;
    bool filterRects = true;
    bool helpScreen = false;

    int detections_num;

    for (;;) {

        ros::spinOnce();

        if(toggle) {

            boost::posix_time::ptime init = boost::posix_time::microsec_clock::local_time();

            ros_grabber.getImage(&frame);

            if (frame.empty())
            {
                continue;
            }

            frame_gpu.upload(image.empty() ? frame : image);

            convertAndResize(frame_gpu, gray_gpu, resized_gpu, scaleFactor);

            TickMeter tm;
            tm.start();

            cascade_gpu.findLargestObject = findLargestObject;

            detections_num = cascade_gpu.detectMultiScale(resized_gpu, facesBuf_gpu, 1.2, (filterRects || findLargestObject) ? 4 : 0);
            facesBuf_gpu.colRange(0, detections_num).download(faces_downloaded);

            resized_gpu.download(resized_cpu);

            if(draw) {
                for (int i = 0; i < detections_num; ++i) {
                   rectangle(resized_cpu, faces_downloaded.ptr<cv::Rect>()[i], Scalar(255,191,0));
                }
            }

            std_msgs::Header h;
            h.stamp = ros_grabber.getTimestamp();
            h.frame_id = "0";

            // ROS MSGS
            people_msgs::People people_msg;
            people_msgs::Person person_msg;
            people_msg.header = h;

            double face_size = 0.0;

            for (int i = 0; i < detections_num; ++i) {
                person_msg.name = "unknown";
                person_msg.reliability = 0.0;
                geometry_msgs::Point p;
                Point center = Point((faces_downloaded.ptr<cv::Rect>()[i].x + faces_downloaded.ptr<cv::Rect>()[i].width/2.0), (faces_downloaded.ptr<cv::Rect>()[i].y + faces_downloaded.ptr<cv::Rect>()[i].height/2.0));
                // double mid_x = center.x;
                // double mid_y = center.y;
                p.x = center.x;
                p.y = center.y;
                p.z = faces_downloaded.ptr<cv::Rect>()[i].size().area();
                face_size = faces_downloaded.ptr<cv::Rect>()[i].size().area();
                person_msg.position = p;
                people_msg.people.push_back(person_msg);
            }

            // TODO revert this, this is just for the stupid Floka
            if (face_size > 4096.0) {
                people_pub.publish(people_msg);
            }

            tm.stop();
            double detectionTime = tm.getTimeMilli();
            double fps = 1000 / detectionTime;

            if(draw) {
                cvtColor(resized_cpu, frameDisp, CV_GRAY2BGR);
                displayState(frameDisp, helpScreen, useGPU, findLargestObject, filterRects, fps);
                imshow(":: CLF GPU Face Detect [ROS] ::", frameDisp);
            }

        }

        char key = (char)waitKey(1);

        if (key == 27)
        {
            break;
        }

        switch (key)
        {
        case 'm':
        case 'M':
            findLargestObject = !findLargestObject;
            break;
        case 'f':
        case 'F':
            filterRects = !filterRects;
            break;
        case '1':
            scaleFactor *= 1.05;
            break;
        case 'q':
        case 'Q':
            scaleFactor /= 1.05;
            break;
        case 'h':
        case 'H':
            // helpScreen = !helpScreen;
            break;
        case 's':
        case 'S':
            draw = !draw;
            break;
        }

        boost::posix_time::ptime c = boost::posix_time::microsec_clock::local_time();
        boost::posix_time::time_duration cdiff = c - init;
        duration = std::to_string(cdiff.total_milliseconds());

        cout << "Computation Time: " << duration << endl;
    }

    return 0;
}
