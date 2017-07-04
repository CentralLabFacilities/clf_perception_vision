#pragma once

// CUDA
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/xfeatures2d/cuda.hpp>
#include <opencv2/cudaobjdetect.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudafilters.hpp>

// CV
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>

// DLIB
#include <dlib/image_processing.h>
#include <dlib/opencv.h>
#include <dlib/image_transforms.h>


class CFaceProcessing
{
private:
   cv:cuda::GpuMat skinBinImg_gpu, skinBinImg_gpu_t, m_grayImg_gpu, skinSegGrayImg_gpu, m_faces_gpu;
   cv::Ptr<cuda::CascadeClassifier> cascade_cuda;
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
   CFaceProcessing(std::string faceXml, std::string eyeXml, std::string glassXml, std::string landmarkDat);
   ~CFaceProcessing();
   int FaceDetection(const cv::Mat colorImg);
   int FaceDetection_GPU(const cv::Mat colorImg);
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