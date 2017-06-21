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

#include <sstream>
#include <mutex>
#include <iostream>
#include "opencv2/opencv.hpp"
#include <boost/lexical_cast.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>

using namespace cv;
using namespace std;

int drag = 0;
int select_flag = 0;

Rect rect;
mutex locker;
Point point1, point2;
Mat img, img1 ,roiImg, toExtract;

const int fontFace = cv::FONT_HERSHEY_PLAIN;
const double fontScale = 1;

void mouseHandler(int event, int x, int y, int flags, void *param)
{

    locker.lock();
    img1 = img.clone();
    locker.unlock();
    imshow(":: CLF GPU Collect Extract Sample ::", img1);

    if (event == CV_EVENT_LBUTTONDOWN && !drag)
    {
        point1 = Point(x, y);
        drag = 1;
    }

    if (event == CV_EVENT_MOUSEMOVE && drag)
    {
        point2 = Point(x, y);
        putText(img1, "Saved ROI", Point2d(point1.x-5, point1.y-5), fontFace, fontScale, CV_RGB(255, 0, 0), 1);
        rectangle(img1, point1, point2, CV_RGB(255, 0, 0), 2, 8, 0);
        imshow(":: CLF GPU Collect Extract Sample ::", img1);
    }

    if (event == CV_EVENT_LBUTTONUP && drag)
    {
        point2 = Point(x, y);
        // Remove the rectangle: +- 3 pixels
        rect = Rect(point1.x+2, point1.y+2, x-2 - point1.x, y-2 - point1.y);
        locker.lock();
        roiImg = img1(rect);
        locker.unlock();
        roiImg.copyTo(toExtract);
        time_t seconds;
        time(&seconds);
        string ts = boost::lexical_cast<std::string>(seconds);
        imwrite(ts+".png", toExtract);
        drag = 0;
    }

    if (event == CV_EVENT_LBUTTONUP)
    {
        select_flag = 1;
        drag = 0;
    }

}

int main()
{

    VideoCapture cap(0);
    if (!cap.isOpened())
      return 1;

    locker.lock();
    cap >> img;
    locker.unlock();

    imshow(":: CLF GPU Collect Live ::", img);

    while (true) {

        locker.lock();
        cap >> img;
        locker.unlock();

        if (img.empty())
            break;

        if (rect.width == 0 && rect.height == 0) {
            cvSetMouseCallback(":: CLF GPU Collect Live ::", mouseHandler, NULL );
        }

        imshow(":: CLF GPU Collect Live ::", img);

        if (cv::waitKey(1) == 27) { break; }
    }

    return 0;
}