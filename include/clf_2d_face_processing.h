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


#pragma once

// STD
#include <iostream>
#include <string>
#include <iostream>
#include <iomanip>

// CUDA
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/xfeatures2d/cuda.hpp>
#include <opencv2/cudaobjdetect.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>

// CV
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>

// DLIB
#include <dlib/image_processing.h>
#include <dlib/opencv.h>
#include <dlib/image_transforms.h>

// BOOST
#include "boost/date_time/posix_time/posix_time.hpp"

class CFaceProcessing
{
private:
   cv::cuda::GpuMat skinBinImg_gpu, skinBinImg_gpu_t, m_grayImg_gpu, skinSegGrayImg_gpu, m_faces_gpu;
   cv::Ptr<cv::cuda::CascadeClassifier> cascade_cuda;
   cv::CascadeClassifier cascade_glasses;
   cv::CascadeClassifier cascade_eyes;
   std::vector<cv::Rect> m_faces;
   // 0 if this face has no enough facial features. Otherwise,
   // this variable indicates how many previous frames the face can be tracked
   std::vector<unsigned char> m_faceStatus;
   std::vector<std::vector<cv::Point> > m_landmarks;
   cv::Mat m_grayImg;
   dlib::shape_predictor m_shapePredictor;
   unsigned int m_normalFaceSize;
   int EyeDetection();
public:  
   CFaceProcessing(std::string faceXml, std::string eyeXml, std::string glassXml, std::string landmarkDat, cv::Size min, cv::Size max, int nei);
   ~CFaceProcessing();
   int FaceDetection_GPU(const cv::Mat colorImg, double scale_fact);
   std::vector<cv::Rect>& GetFaces();   
   int AlignFaces2D(std::vector<cv::Mat>& alignedFaces, cv::Mat originalbool, bool onlyLargest = false);
   int GetLargestFace();
   void FaceHistogramEqualization(cv::Mat& faceImg);
   std::vector<cv::Point>& GetLandmarks(const unsigned int idx);
   cv::Mat& GetGrayImages();
   int FindLandmarksWhichFaces(const std::vector<cv::Point2f>::iterator& landmark, const int n);
   std::vector<unsigned char> GetFaceStatus();
   bool IncFaceStatus(const int idx, const int val);
   void CleanFaces();
};