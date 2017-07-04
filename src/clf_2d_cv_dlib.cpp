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
#include "clf_2d_caffee_classification.h"
#include "clf_2d_face_processing.h"

// ROS
#include <ros/ros.h>
#include <std_msgs/Bool.h>
#include <people_msgs/People.h>
#include <people_msgs/Person.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/PointStamped.h>

using namespace std;
using namespace cv;
using namespace cuda;
using namespace dlib;

bool detect_gender = false;
bool toggle = true;
bool draw = true;

unsigned int average_frames = 0;
unsigned int last_computed_frame = -1;
unsigned int frame_count = 0;
// Default
unsigned int min_n = 4;

// Dynamic
double scaleFactor = 1.4;
double time_spend = 0;

bool findLargestObject = true;

// Defaults
Size minSize(80,80);
Size maxSize(400,400);

const int fontFace = cv::FONT_HERSHEY_PLAIN;
const double fontScale = 1;

string cascade_frontal_file,
       cascade_profile_file,
       cascade_nose_file,
       cascade_mouth_file,
       dlib_shapepredictor,
       topic,
       model_file,
       trained_file,
       mean_file,
       label_file;

static void help()
{
    cout << "Usage: <proc> \n\t config.yaml\n" << endl;
}

static void matPrint(Mat &img, int lineOffsY, Scalar fontColor, const string &ss)
{
    int fontFace = FONT_HERSHEY_DUPLEX;
    double fontScale = 0.45;
    int fontThickness = 0.3;
    Size fontSize = getTextSize("T[]", fontFace, fontScale, fontThickness, 0);

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

    ros::init(argc, argv, "clf_detect_faces", ros::init_options::AnonymousName);
    ros::NodeHandle nh_;
    ros::Publisher people_pub;
    ros::Subscriber toggle_sub;

    people_pub = nh_.advertise<people_msgs::People>("clf_detect_faces/people", 20);
    toggle_sub = nh_.subscribe("/clf_detect_faces/compute", 1, toggle_callback);

    if (getCudaEnabledDeviceCount() == 0)
    {
        return cerr << ">>> No GPU found or OpenCV is compiled without GPU support" << endl, -1;
    }

    cout << ">>> Cuda Enabled Devices --> " << cuda::getCudaEnabledDeviceCount() << endl;
    cout << ">>> ";
    cuda::printShortCudaDeviceInfo(cuda::getDevice());

    CommandLineParser parser(argc, argv, "{@config |<none>| yaml config file}" "{help h ||}");
    FileStorage fs(argv[1], FileStorage::READ);

    if (fs.isOpened()) {

        fs["input_ros_topic"] >> topic;
        cout << ">>> Input Topic: --> " << topic << endl;

        fs["cascade_frontal_file"] >> cascade_frontal_file;
        cout << ">>> Frontal Face: --> " << cascade_frontal_file << endl;

        fs["cascade_profile_file"] >> cascade_profile_file;
        cout << ">>> Profile Face: --> " << cascade_profile_file << endl;

        fs["cascade_nose_file"] >> cascade_nose_file;
        cout << ">>> Nose: --> " << cascade_nose_file << endl;

        fs["cascade_mouth_file"] >> cascade_mouth_file;
        cout << ">>> Mouth: --> " << cascade_mouth_file << endl;

        fs["dlib_shapepredictor"] >> dlib_shapepredictor;
        cout << ">>> DLIB: --> " << dlib_shapepredictor << endl;

        fs["model_file"] >> model_file;
        cout << ">>> Caffee Model: --> " << model_file << endl;

        fs["trained_file"] >> trained_file;
        cout << ">>> Caffee Trained: --> " << trained_file << endl;

        fs["mean_file"] >> mean_file;
        cout << ">>> Caffe Mean: --> " << mean_file << endl;

        fs["label_file"] >> label_file;
        cout << ">>> Labels: --> " << label_file << endl;
    }

    fs.release();

    // ROS
    ROSGrabber ros_grabber(topic);

    // Caffee
    Classifier classifier(model_file, trained_file, mean_file, label_file);

    // Faces
    CFaceProcessing fp(cascade_frontal_file, cascade_nose_file, cascade_mouth_file, dlib_shapepredictor, minSize, maxSize, min_n);

    namedWindow(":: CLF GPU Face Detect [ROS] Press ESC to Exit ::", 1);

    Mat frame, frame_display;

    std::vector<std::vector<cv::Point> > fLandmarks;

    time_t start, end;
    time(&start);

    while(waitKey(2) != 27) {

        ros::spinOnce();

        if(toggle) {
            ros_grabber.getImage(&frame);
            if (frame.rows*frame.cols > 0) {
                int tmp_frame_nr = ros_grabber.getLastFrameNr();
                if(last_computed_frame != tmp_frame_nr) {
                    frame_display = frame.clone();
                    std::vector<Rect> faces;
                    if(draw) {
                         int faceNum ;
                         faceNum = fp.FaceDetection_GPU(frame, scaleFactor);
                         std::vector<cv::Mat> croppedImgs;
                         if (faceNum > 0)
                         {
                            faces = fp.GetFaces();
                            // normalize the face image with landmark
                            std::vector<cv::Mat> normalizedImg;
                            fp.AlignFaces2D(normalizedImg, frame);
                            croppedImgs.resize(faceNum);
                            for (int i = 0; i < faceNum; i++)
                            {
                               // ------------------------------
                               // Sylar 20160308 to use RGBscale
                               // ------------------------------
                               int x = faces[i].x - (faces[i].width / 4);
                               int y = faces[i].y - (faces[i].height / 4);
                               if (x < 0)
                                  x = 0;
                               if (y < 0)
                                  y = 0;
                               int w = faces[i].width + (faces[i].width / 2) ;
                               int h = faces[i].height + (faces[i].height / 2);
                               if(w + x > frame.cols)
                                  w = frame.cols - x ;
                               if(h + y > frame.rows)
                                  h = frame.rows - y ;
                               croppedImgs[i] = frame(cv::Rect(x, y, w, h)).clone();
                            }
                            // ---------------------------------
                            // extraction landmarks on each face
                            // ---------------------------------
                             fLandmarks.resize(faceNum);

                             for (int i = 0; i < faceNum; i++)
                             {
                                fLandmarks[i] = fp.GetLandmarks(i);
                             for (size_t j = 0; j < fLandmarks[i].size(); j++)
                                cv::circle(frame_display, fLandmarks[i][j], 1, cv::Scalar(255,255,255));
                             }
                         }
                         // --------------------------------------------
                         // do gender classification and display results
                         // --------------------------------------------
                         std::vector<unsigned char> status = fp.GetFaceStatus();

                         for (int i = 0; i < faceNum; i++)
                         {
                            if (status[i])
                            {
                               // cv::imshow("Cropped Images", croppedImgs[i]);
                               // cv::waitKey(2);
                               std::vector<Prediction> predictions = classifier.Classify(croppedImgs[i]);
                               Prediction p = predictions[0];
                               if (p.second >= 0.7)
                               {
                                  if (p.first == "male")
                                  {
                                     // char beliefStr[64] = { 0 };
                                     cv::putText(frame_display, p.first, cv::Point(faces[i].x, faces[i].y + faces[i].height + 20), fontFace, fontScale, CV_RGB(70,130,180));
                                     cv::rectangle(frame_display, faces[i], CV_RGB(70,130,180), 3);
                                  }
                                  else if(p.first == "female")
                                  {
                                     // char beliefStr[64] = { 0 };
                                     cv::putText(frame_display, p.first, cv::Point(faces[i].x, faces[i].y + faces[i].height + 20), fontFace, fontScale, CV_RGB(221,160,221));
                                     cv::rectangle(frame_display, faces[i], CV_RGB(221,160,221), 3);
                                  }
                               }
                            }
                         }

                         fp.CleanFaces();
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
//                    } else if (faces_profile.size() > 0) {
//                        for (int i = 0; i < faces_profile.size(); ++i) {
//                            person_msg.name = "unknown";
//                            person_msg.reliability = 0.0;
//                            geometry_msgs::Point p;
//                            Point center = Point(faces_profile[i].x + faces_profile[i].width/2.0, faces_profile[i].y + faces_profile[i].height/2.0);
//                            double mid_x = center.x;
//                            double mid_y = center.y;
//                            p.x = center.x;
//                            p.y = center.y;
//                            p.z = faces_profile[i].size().area();
//                            person_msg.position = p;
//                            people_msg.people.push_back(person_msg);
//                        }
                    }

                    people_pub.publish(people_msg);

                    frame_count++;

                    if(draw) {
                        displayState(frame_display, scaleFactor);
                        imshow(":: CLF GPU Face Detect [ROS] Press ESC to Exit ::", frame_display);
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

        char key = (char)waitKey(2);

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
        }

    }
    return 0;
}
