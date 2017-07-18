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


#include "clf_2d_face_processing.h"

using namespace cv;
using namespace cuda;

CFaceProcessing::CFaceProcessing(std::string faceXml, std::string eyeXml, std::string glassXml, std::string landmarkDat, cv::Size min, cv::Size max, int nei) : m_normalFaceSize(128)
{

   if (!cascade_eyes.load(glassXml))
   {
      printf(">>> Error: cannot load xml file for eye(glass) detection in function CFaceProcessing::CFaceProcessing()\n");
   }

   if (!cascade_glasses.load(eyeXml))
   {
      printf(">>> Error: cannot load xml file for eye detection in function CFaceProcessing::CFaceProcessing()\n");
   }

   cascade_cuda = cuda::CascadeClassifier::create(faceXml);
   cascade_cuda->setMinNeighbors(nei);
   cascade_cuda->setScaleFactor(1.4);
   cascade_cuda->setFindLargestObject(false);
   cascade_cuda->setMinObjectSize(min);
   cascade_cuda->setMaxObjectSize(max);

   dlib::deserialize(landmarkDat) >> m_shapePredictor;
}

CFaceProcessing::~CFaceProcessing() { }

int CFaceProcessing::FaceDetection_GPU(const cv::Mat colorImg, double scale_fact, bool pyr)
{
   // dynamic settings
   cascade_cuda->setScaleFactor(scale_fact);

   // color space conversion
   cv::Mat yCbCrImg;
   cv::cvtColor(colorImg, yCbCrImg, CV_RGB2YCrCb);

   // copy first channel of image in YCbCr to gray image 
   m_grayImg = cv::Mat(yCbCrImg.size(), CV_8UC1);
   int fromTo[] = { 0, 0 };
   cv::mixChannels(&yCbCrImg, 1, &m_grayImg, 1, &fromTo[0], 1);

   // segmentation for skin color
   cv::Mat skinBinImg;
   cv::inRange(yCbCrImg, cv::Scalar(0, 85, 135), cv::Scalar(255, 135, 180), skinBinImg);

   // convert to GpuMat
   skinBinImg_gpu.upload(skinBinImg);

   // erode and dilate to remove small segmentation
   cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));

   cv::Ptr<cv::cuda::Filter> erode = cv::cuda::createMorphologyFilter(cv::MORPH_ERODE, skinBinImg_gpu.type(), kernel);
   erode->apply(skinBinImg_gpu, skinBinImg_gpu_t);
   cv::Ptr<cuda::Filter> dilateFilter = cv::cuda::createMorphologyFilter(cv::MORPH_ERODE, skinBinImg_gpu_t.type(), kernel);
   dilateFilter->apply(skinBinImg_gpu_t, skinBinImg_gpu);

   // apply GaussianBlur to have a complete segmentation
   cv::Ptr<cv::cuda::Filter> blur = cv::cuda::createGaussianFilter(skinBinImg_gpu.type(), skinBinImg_gpu_t.type(), cv::Size(0, 0), 3.0);
   blur->apply(skinBinImg_gpu, skinBinImg_gpu_t);

   // -----------------------------------------------
   // face detection with OpenCV on skin-color region
   // -----------------------------------------------
   std::vector<Rect> faces;
   m_grayImg_gpu.upload(m_grayImg);
   m_grayImg_gpu.copyTo(skinSegGrayImg_gpu, skinBinImg_gpu_t);

   cascade_cuda->detectMultiScale(skinSegGrayImg_gpu, m_faces_gpu);
   cascade_cuda->convert(m_faces_gpu, faces);

   for(int i = 0; i < faces.size(); ++i) {
      // cv::rectangle(m_grayImg, faces[i], cv::Scalar(255));
      m_faces.push_back(faces[i]);
   }

   // START remove if you want eye detection
   m_faceStatus.resize(m_faces.size(), 0);

   for(int i = 0; i < m_faces.size(); ++i) {
      m_faceStatus[i] = 1;
   }
   // END remove if you want eye detection

   // eye detection
   // EyeDetection();

   return m_faces.size();
}

std::vector<cv::Rect>& CFaceProcessing::GetFaces()
{
   return m_faces;
}

int CFaceProcessing::EyeDetection()
{
   // before calling this function, make sure function "FaceDetection" has been called
   m_faceStatus.resize(m_faces.size(), 0);

   for (unsigned int i = 0; i < m_faces.size() || i < 0; i++)
   {
      cv::Mat faceImg;
      m_grayImg(m_faces[i]).copyTo(faceImg);
      // histogram equalization on face
      cv::equalizeHist(faceImg, faceImg);
      std::vector<cv::Rect> faceFeature;
      cascade_eyes.detectMultiScale(faceImg, faceFeature, 1.2, 4, CV_HAAR_FIND_BIGGEST_OBJECT, cv::Size(4, 4));
      if (faceFeature.size() != 0)
      {
         cascade_glasses.detectMultiScale(faceImg, faceFeature, 1.2, 4, CV_HAAR_FIND_BIGGEST_OBJECT, cv::Size(4, 4));
         if (faceFeature.size() != 0) m_faceStatus[i] = 1;
         else m_faceStatus[i] = 0;
      }
      else
      {
         m_faceStatus[i] = 0;
      }
   }
   return m_faces.size();
}

int CFaceProcessing::AlignFaces2D(std::vector<cv::Mat>& alignedFaces, cv::Mat original, bool onlyLargest)
{
   // before calling this function, make sure function "FaceDetection" has been called
   std::vector<cv::Rect> faces;
   
   // find the largest face
   if (onlyLargest == true)
   {
      int idx = GetLargestFace();
      if (idx >= 0) faces.push_back(m_faces[idx]);
   }
   else faces = m_faces;  

   // landmark detection on faces
   std::vector<dlib::full_object_detection> shapes;
   shapes.resize(faces.size());
   dlib::cv_image<unsigned char> dlib_img(m_grayImg);
   m_landmarks.resize(faces.size());
   for (int i = 0; i < (int)faces.size(); i++)
   {
      int x = faces[i].x - 30;
      int y = faces[i].y - 30;
      if (x < 0)
         x = 0;
      if (y < 0)
         y = 0;
      int w = faces[i].width + 60;
      int h = faces[i].height + 60;
      if(w + faces[i].x > original.cols)
         w = original.cols - faces[i].x ;
      if(h + faces[i].y > original.rows)
         h = original.rows - faces[i].y ;
      shapes[i] = m_shapePredictor(dlib_img, dlib::rectangle(x, y, x + w, y + h));
      int partsNum = shapes[i].num_parts();
      m_landmarks[i].resize(partsNum);
      for (int j = 0; j < partsNum; j++)
      {
         m_landmarks[i][j].x = (shapes[i].part(j)).x();
         m_landmarks[i][j].y = (shapes[i].part(j)).y();
      }
   }

   // normalize the size of faces
   alignedFaces.resize(faces.size());
   dlib::array<dlib::array2d<unsigned char> > faceChips;
   dlib::extract_image_chips(dlib_img, dlib::get_face_chip_details(shapes, m_normalFaceSize), faceChips);

   for (unsigned int i = 0; i < faces.size(); i++)
   {
      dlib::toMat(faceChips[i]).copyTo(alignedFaces[i]);
   }
   return alignedFaces.size();
}

int CFaceProcessing::GetLargestFace()
{
   int largestIdx = -1;
   int largestArea = 0;
   for (unsigned int i = 0; i < m_faces.size(); i++)
   {
      if (!m_faceStatus[i]) continue;
      int area = m_faces[i].width * m_faces[i].height;
      if (largestArea < area)
      {
         largestIdx = i;
         largestArea = area;
      }
   }
   return largestIdx;
}

std::vector<cv::Point>& CFaceProcessing::GetLandmarks(const unsigned int idx)
{
   // must make sure the idx is valid by yourself before calling this function
   return m_landmarks[idx];
}

cv::Mat& CFaceProcessing::GetGrayImages()
{
   return m_grayImg;
}

int CFaceProcessing::FindLandmarksWhichFaces(const std::vector<cv::Point2f>::iterator& landmark, const int n)
{
   int faceIdx = -1;
   for (unsigned int i = 0; i < m_faces.size(); i++)
   {
      int vote = 0;
      for (int j = 0; j < n; j++)
      {
         cv::Point pt = *(landmark + j);
         if (m_faces[i].contains(pt) == true) vote++;
      }

      if (vote >= (n >> 1)) // (TBD) 1/2 landmarks must be tracked for now
      {
         faceIdx = (int)i;
         break;
      }
   }
   return faceIdx;
}

std::vector<unsigned char> CFaceProcessing::GetFaceStatus()
{
   return m_faceStatus;
}

bool CFaceProcessing::IncFaceStatus(const int idx, const int val)
{
   if (m_faceStatus.size() < idx) return false;
   m_faceStatus[idx] += val;
   if (m_faceStatus[idx] > 200) m_faceStatus[idx] = 200;
   return true;
}

void CFaceProcessing::CleanFaces(){
   m_faces.clear();
}
