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


// STD
#include <iostream>
#include <string>

// THREADING
#include <thread>
#include <mutex>

// SELF
#include "clf_2d_detect.hpp"
#include "ros_grabber.hpp"

using namespace std;

int main(int argc, char *argv[]) {

    if (argc < 4) {
        cout << ">>> Usage: clf_2d_detect {path/to/config/file} {input scope} {loop rate}" << endl;
        cout << ">>> Example: clf_2d_detect /tmp/example.yaml /usb_cam/image_raw 50" << endl;
        return -1;
    }

    ros::init(argc, argv, "clf_2d_detect", ros::init_options::AnonymousName);

    ROSGrabber ros_grabber(argv[2]);
    Detect2D detect2d;

    detect2d.setup(argc, argv);

    cv::namedWindow(":: CLF GPU Detect [ROS] ::", cv::WINDOW_AUTOSIZE);
    cv::Mat current_image;

    ros::Rate loop_rate(atoi(argv[3]));

    while(cv::waitKey(1) <= 0){
        ros::spinOnce();
        boost::posix_time::ptime start_main = boost::posix_time::microsec_clock::local_time();
        ros::Time frame_timestamp;
        try {
            ros_grabber.getImage(&frame_timestamp, &current_image);
            if (current_image.rows+current_image.cols > 0) {
                detect2d.detect(current_image, ros_grabber.getDuration());
            } else {
                cout << "E >>> Image could not be grabbed" << endl;
            }
        } catch (std::exception& e) {
            cout << "E >>> " << e.what() << endl;
        }

        boost::posix_time::ptime end_main = boost::posix_time::microsec_clock::local_time();
        boost::posix_time::time_duration diff_main = end_main - start_main;
        string string_time_main = to_string(diff_main.total_milliseconds());

        cv::putText(current_image, "Delta T (Total): "+string_time_main+" ms", cv::Point2d(current_image.cols-220, 80), detect2d.fontFace, detect2d.fontScale, cv::Scalar(219, 152, 52), 1);
        cv::imshow(":: CLF GPU Detect [ROS] ::", current_image);

        loop_rate.sleep();
    }

    ros_grabber.closeGrabber();
    cv::destroyAllWindows();
}

