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
#include "native_grabber.hpp"

NativeGrabber::NativeGrabber(){}
NativeGrabber::~NativeGrabber(){}

void NativeGrabber::setup(int i_width, int i_height, int cam) {
    height = i_height;
    width = i_width;

    cv::VideoCapture capture(cam);
    cap = capture;
    cap.set(CV_CAP_PROP_FRAME_WIDTH, width);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, height);

    if (!cap.isOpened()) {
        std::cout << "E >>> Camera cannot be opened" << std::endl;
        exit(EXIT_FAILURE);
    }

    is_setup = true;
}

void NativeGrabber::grabImage(cv::Mat *input_image) {
    boost::posix_time::ptime start = boost::posix_time::microsec_clock::local_time();

    if (cap.grab()) {
        if (!cap.retrieve(source_frame)) {
            std::cout << "E >>> Failed to retrieve image, DROPPED ONE FRAME!" << std::endl;
        }

        if (source_frame.rows*source_frame.cols <= 0) {
            std::cout << "E >>> Camera Image " << " is NULL" << std::endl;
        }

        *input_image = source_frame;
    }

    boost::posix_time::ptime end = boost::posix_time::microsec_clock::local_time();
    boost::posix_time::time_duration diff = end - start;
    duration = std::to_string(diff.total_milliseconds());
}

bool NativeGrabber::isSetup() {
    return is_setup;
}

void NativeGrabber::closeGrabber(){
    cap.release();
    std::cout << ">>> Cleaning Up. Goodbye!" << std::endl;
}

std::string NativeGrabber::getDuration() {
    return duration;
}
