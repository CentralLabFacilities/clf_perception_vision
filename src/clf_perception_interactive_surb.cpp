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


// ROS
#include <std_msgs/Header.h>
#include <geometry_msgs/Pose.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

// SELF
#include "clf_perception_interactive_surb.hpp"


using namespace std;
using namespace cv;

Detect2DInteractive::Detect2DInteractive(){}
Detect2DInteractive::~Detect2DInteractive(){}

time_t t = time(nullptr);
char *t_c = asctime(localtime(&t));
RNG rng(133742+t);

const int minhessian = 400;

string out_topic = "/clf_perception_interactive_surb/objects";

Ptr<cuda::DescriptorMatcher> cuda_bf_matcher = cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);
Ptr<cuda::DescriptorMatcher> cuda_knn_matcher = cuda::DescriptorMatcher::createBFMatcher(cv::NORM_L1);

// See: http://docs.opencv.org/2.4/modules/nonfree/doc/feature_detection.html
cuda::SURF_CUDA cuda_surf(minhessian, 4, 2, true, true);

vector<Scalar> Detect2DInteractive::color_mix(int count) {
    vector<Scalar> colormix;
    for (int i=0; i<count; i++) {
        Scalar color = Scalar(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255));
        colormix.push_back(color);
    }
    return colormix;
}

bool is_file_exist(const char *fileName)
{
    ifstream infile(fileName);
    return infile.good();
}

int Detect2DInteractive::get_x_resolution() {
    return res_x;
}

int Detect2DInteractive::get_y_resolution() {
    return res_y;
}

bool Detect2DInteractive::get_silent() {
    return toggle_silent;
}

int handleError(int status, const char* func_name,
                const char* err_msg, const char* file_name,
                int line, void* userdata) {
    return 0;
}

int Detect2DInteractive::setup(int argc, char *argv[]) {

    cout << ">>> Cuda Enabled Devices --> " << cuda::getCudaEnabledDeviceCount() << endl;

    if (cuda::getCudaEnabledDeviceCount() == 0)
    {
        cout << "E >>> No cuda Enabled Devices" << endl;
        exit(EXIT_FAILURE);
    } else {
        cout << ">>> ";
        cuda::printShortCudaDeviceInfo(cuda::getDevice());
    }

    if (!is_file_exist(argv[1])) {
        cout << "E >>> File does not exist --> " << argv[1]  << endl;
        exit(EXIT_FAILURE);
    }

    cout << ">>> ROS Out Topic --> " << out_topic << endl;

    CommandLineParser parser(argc, argv, "{@config |<none>| yaml config file}" "{help h ||}");
    FileStorage fs(argv[1], FileStorage::READ);

    if (fs.isOpened()) {

        fs["input_ros_topic"] >> ros_input_topic;
        cout << ">>> ROS input topic --> " << ros_input_topic  << endl;

        fs["keypointalgo"] >> type_descriptor;
        cout << ">>> Keypoint Descriptor --> " << type_descriptor  << endl;

        fs["matcher"] >> point_matcher;
        cout << ">>> Matching Algorithm --> " << point_matcher  << endl;

        fs["maxkeypoints_orb"] >> max_keypoints;
        cout << ">>> Max Key Points (ORB ONLY): --> " << max_keypoints  << endl;

        fs["minmatches"] >> min_matches;
        cout << ">>> Min Matches: --> " << min_matches  << endl;

        fs["maxmatches"] >> max_matches;
        cout << ">>> Max Matches: --> " << max_matches  << endl;

        detection_threshold = fs["detectionthreshold"];
        cout << ">>> Detection Threshold --> " << detection_threshold << endl;

        pyr = (int) fs["pyr_up"];

        cout << ">>> PYRUP in config: --> " << pyr << endl;

        if (pyr > 0) {
            cout << ">>> Image Scaling is: --> ON" << endl;
            scale_factor = 2.0;
            cout << ">>> Scale Factor: --> " << scale_factor << endl;
        } else {
            cout << ">>> Image Scaling is: --> OFF" << endl;
        }

        toggle_homography = true;

        fs["silent"] >> draw_image;
        cout << ">>> Silent --> " << draw_image << endl;

        if (draw_image == "true") {
            toggle_silent = true;
        }
    }

    fs.release();

    if (type_descriptor.compare("ORB") == 0) {
        // See init values: http://docs.opencv.org/trunk/da/d44/classcv_1_1cuda_1_1ORB.html
        cuda_orb = cuda::ORB::create(max_keypoints,
                                     1.2f,
                                     8,
                                     31,
                                     0,
                                     2,
                                     cuda::ORB::HARRIS_SCORE,
                                     31,
                                     20,
                                     true);
    } else if (type_descriptor.compare("SURF") == 0) {
        // OK
    } else {
        cout << "E >>> Unknown Detector Algorithm " << type_descriptor << endl;
        exit(EXIT_FAILURE);
    }

    cout << ">>> Initialized ---> " << type_descriptor << " Algorithm" << endl;

    object_pub = node_handle_.advertise<visualization_msgs::MarkerArray>(out_topic, 1);

    return 0;

}

bool Detect2DInteractive::addTarget(clf_perception_vision_msgs::LearnPersonImage::Request &req, clf_perception_vision_msgs::LearnPersonImage::Response&res) {

    mtx.lock();

    cv_bridge::CvImagePtr cv_ptr;

    try {
        cv_ptr = cv_bridge::toCvCopy(req.roi, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception &e) {
        res.success = false;
        res.name = req.name;
        ROS_ERROR("E >>> CV_BRIDGE exception: %s", e.what());
        mtx.unlock();
        return false;
    }

    cv::Mat img_grey;
    cv::Mat output_frame_grey;

    cv::Mat source_frame = cv_ptr->image;
    cv::cvtColor(source_frame, img_grey, CV_BGR2GRAY);
    cv::pyrUp(img_grey, output_frame_grey, cv::Size(source_frame.cols*2, source_frame.rows*2));

    cuda::GpuMat cuda_tmp_img(output_frame_grey);

    try {

        if (cuda_tmp_img.rows * cuda_tmp_img.cols <= 0) {
            cout << "E >>> CUDA Image is empty or cannot be found" << endl;
            res.success = false;
            res.name = req.name;
            mtx.unlock();
            return false;
        }

        vector <KeyPoint> tmp_kp;
        cuda::GpuMat tmp_cuda_dc;

        if (type_descriptor.compare("ORB") == 0) {
            try {
                cuda_orb->detectAndCompute(cuda_tmp_img, cuda::GpuMat(), tmp_kp, tmp_cuda_dc);
            }
            catch (Exception &e) {
                cout << "E >>> ORB init fail O_O | Maybe not enough key points in training image" << "\n";
                res.success = false;
                res.name = req.name;
                mtx.unlock();
                return false;
            }
        }

        if (type_descriptor.compare("SURF") == 0) {
            try {
                cuda_surf(cuda_tmp_img, cuda::GpuMat(), tmp_kp, tmp_cuda_dc);
            }
            catch (Exception &e) {
                cout << "E >>> SURF init fail O_O | Maybe not enough key points in training image" << "\n";
                res.success = false;
                res.name = req.name;
                mtx.unlock();
                return false;
            }
        }

        target_labels.push_back(req.name);
        keys_current_target.push_back(tmp_kp);
        target_images.push_back(output_frame_grey);
        cuda_desc_current_target_image.push_back(tmp_cuda_dc);

    } catch (Exception &e) {
          ROS_ERROR("E >>> CV_BRIDGE exception: %s", e.what());
          res.success = false;
          res.name = req.name;
          target_labels.clear();
          keys_current_target.clear();
          target_images.clear();
          cuda_desc_current_target_image.clear();
          mtx.unlock();
          return false;
    }

    // Generate Random Colors
    colors = color_mix(target_images.size());
    res.success = true;
    res.name = req.name;

    // cout << ">>> Current key points size: --> " << keys_current_target.size() << endl;
    // cout << ">>> Current target images size --> " << target_images.size() << endl;
    // cout << ">>> Current cuda descriptors size --> " << target_images.size() << endl;

    mtx.unlock();

    return true;
}


void Detect2DInteractive::detect(Mat input_image, ros::Time timestamp, std::string frame_id) {

    if (target_images.size() > 0) {

        mtx.lock();

        // START keypoint extraction in actual image //////////////////////////////////////////
        visualization_msgs::MarkerArray ma;

        boost::posix_time::ptime start_detect = boost::posix_time::microsec_clock::local_time();
        cuda_frame_tmp_img.upload(input_image);

        if (scale_factor > 1.0) {
            cuda::pyrUp(cuda_frame_tmp_img, cuda_frame_scaled);
            cuda::cvtColor(cuda_frame_scaled, cuda_camera_tmp_img, COLOR_BGR2GRAY);
        } else {
            cuda::cvtColor(cuda_frame_tmp_img, cuda_camera_tmp_img, COLOR_BGR2GRAY);
        }

        if (type_descriptor.compare("ORB") == 0) {
            try {
                cuda_orb->detectAndCompute(cuda_camera_tmp_img, cuda::GpuMat(), keys_camera_image,
                                           cuda_desc_camera_image);
            }
            catch (Exception &e) {
                cout << "E >>> ORB fail O_O" << "\n";
            }
        }

        if (type_descriptor.compare("SURF") == 0) {
            try {
                cuda_surf(cuda_camera_tmp_img, cuda::GpuMat(), keys_camera_image, cuda_desc_camera_image);
            }
            catch (Exception &e) {
                cout << "E >>> SURF fail O_O" << "\n";
            }
        }

        boost::posix_time::ptime end_detect = boost::posix_time::microsec_clock::local_time();

        if (keys_camera_image.empty()) {
            return;
        }
        // END keypoint extraction in actual image //////////////////////////////////////////

        // START keypoint matching: actual image and saved descriptor////////////////////////
        vector <vector<DMatch>> cum_best_matches;
        boost::posix_time::ptime start_match = boost::posix_time::microsec_clock::local_time();

        for (unsigned int i = 0; i < target_images.size(); i++) {

            vector <DMatch> matches;
            vector <vector<DMatch>> knn_matches;
            vector <DMatch> bestMatches;

            try {
                try {
                    if (!cuda_desc_current_target_image[i].empty() && !cuda_desc_camera_image.empty()) {

                        if (point_matcher.compare("BF") == 0) {

                            cuda_bf_matcher->match(cuda_desc_current_target_image[i], cuda_desc_camera_image, matches);

                            Mat index;
                            int nbMatch = int(matches.size());

                            Mat tab(nbMatch, 1, CV_32F);

                            for (int i = 0; i < nbMatch; i++) {
                                tab.at<float>(i, 0) = matches[i].distance;
                            }

                            sortIdx(tab, index, SORT_EVERY_COLUMN + SORT_ASCENDING);

                            for (int i = 0; i < (int) matches.size() - 1; i++) {
                                if (matches[index.at<int>(i, 0)].distance <
                                    detection_threshold * matches[index.at<int>(i + 1, 0)].distance) {
                                    bestMatches.push_back(matches[index.at<int>(i, 0)]);
                                    if (i >= max_matches) {
                                        break;
                                    }
                                }
                            }

                            cum_best_matches.push_back(bestMatches);
                        }

                        if (point_matcher.compare("KNN") == 0) {

                            cuda_knn_matcher->knnMatch(cuda_desc_current_target_image[i], cuda_desc_camera_image,
                                                       knn_matches, 2);

                            for (int k = 0; k < std::min(keys_camera_image.size() - 1, knn_matches.size()); k++) {
                                if ((knn_matches[k][0].distance < detection_threshold * (knn_matches[k][1].distance)) &&
                                    ((int) knn_matches[k].size() <= 2 && (int) knn_matches[k].size() > 0)) {
                                    bestMatches.push_back(knn_matches[k][0]);
                                    if (k >= max_matches) {
                                        break;
                                    }
                                }
                            }

                            cum_best_matches.push_back(bestMatches);
                        }

                    } else {
                        do_not_draw = true;
                        cout << "E >>> Descriptors are empty" << endl;
                    }
                } catch (Exception &e) {
                    cout << "E >>> Cumulative distance cannot be computed, next iteration" << endl;
                    continue;
                }
            } catch (Exception &e) {
                cout << "E >>> Matcher is wating for input..." << endl;
                continue;
            }
        }

        boost::posix_time::ptime end_match = boost::posix_time::microsec_clock::local_time();
        // END keypoint matching: actual image and saved descriptor////////////////////////

        // START fitting //////////////////////////////////////////////////////////////////
        text_offset_y = 20;
        boost::posix_time::ptime start_fitting = boost::posix_time::microsec_clock::local_time();

        if (do_not_draw == false) {
            for (int i = 0; i < target_images.size(); i++) {
                try {

                    vector<DMatch>::iterator it;

                    for (it = cum_best_matches[i].begin(); it != cum_best_matches[i].end(); it++) {
                        Point2d c_t = keys_camera_image[it->trainIdx].pt;
                        Point2d current_point(c_t.x, c_t.y);
                        Point2d current_point_draw(c_t.x / scale_factor, c_t.y / scale_factor);
                        circle(input_image, current_point_draw, 3.0, colors[i], 1, 1);
                    }

                } catch (Exception &e) {
                    cout << "E >>> Could not derive median" << endl;
                    continue;
                }
            }
        }

        unsigned int detected_classes = 0;

        for (unsigned int i = 0; i < target_images.size(); i++) {
            if (toggle_homography) {
                try {

                    if ((int) cum_best_matches[i].size() < min_matches) {
                        // cout << "E >>> Not enough BEST matches: " << cum_best_matches[i].size() << " | " << min_matches << " are required" << endl;
                        continue;
                    }

                    vector <Point2f> obj;
                    vector <Point2f> scene;
                    vector<DMatch>::iterator it;

                    for (it = cum_best_matches[i].begin(); it != cum_best_matches[i].end(); it++) {
                        obj.push_back(keys_current_target[i][it->queryIdx].pt);
                        scene.push_back(keys_camera_image[it->trainIdx].pt);
                    }

                    if (!obj.empty() && !scene.empty()) {

                        // CV_LMEDS --> slower
                        Mat H = findHomography(obj, scene, CV_RANSAC);

                        if (H.empty()) {
                            continue;
                        }

                        double quality = determinant(H);

                        if (quality <= 0.195) {
                            continue;
                        }

                        vector <cv::Point2f> obj_corners(4);
                        obj_corners[0] = Point(0, 0);
                        obj_corners[1] = Point(target_images[i].cols, 0);
                        obj_corners[2] = Point(target_images[i].cols, target_images[i].rows);
                        obj_corners[3] = Point(0, target_images[i].rows);

                        vector <Point2f> scene_corners;
                        vector <Point2f> scene_corners_draw;

                        redirectError(handleError);
                        perspectiveTransform(obj_corners, scene_corners, H);
                        redirectError(nullptr);

                        for (size_t i = 0; i < scene_corners.size(); i++) {
                            float x = scene_corners[i].x / scale_factor;
                            float y = scene_corners[i].y / scale_factor;
                            scene_corners_draw.push_back(cv::Point2d(x, y));
                        }

                        int mid_x = (scene_corners_draw[0].x + scene_corners_draw[2].x) / 2;
                        int mid_y = (scene_corners_draw[0].y + scene_corners_draw[3].y) / 2;
                        int width_x = cv::norm(scene_corners_draw[0] - scene_corners_draw[1]);
                        int width_y = cv::norm(scene_corners_draw[0] - scene_corners_draw[2]);

                        std_msgs::Header h;
                        visualization_msgs::Marker m;

                        geometry_msgs::Pose pose;
                        geometry_msgs::Point pt;

                        h.stamp = timestamp;
                        h.frame_id = frame_id;
                        m.header = h;

                        m.text = target_labels[i];
                        m.ns = target_labels[i];

                        pt.x = mid_x;
                        pt.y = mid_y;
                        pt.z = width_x * width_y;
                        m.pose.position = pt;

                        ma.markers.push_back(m);

                        putText(input_image, target_labels[i], cv::Point2d(mid_x, mid_y), cv::FONT_HERSHEY_PLAIN, 1,
                                colors[i], 2);

                        line(input_image, scene_corners_draw[0], scene_corners_draw[1], colors[i], 4);
                        line(input_image, scene_corners_draw[1], scene_corners_draw[2], colors[i], 4);
                        line(input_image, scene_corners_draw[2], scene_corners_draw[3], colors[i], 4);
                        line(input_image, scene_corners_draw[3], scene_corners_draw[0], colors[i], 4);

                        detected_classes++;
                    }
                } catch (cv::Exception &e) {
                    cout << "WARNING >>> Could not derive perspective transform" << endl;
                }
            }
        }

        if (detected_classes > 0) {
            object_pub.publish(ma);
        }

        boost::posix_time::ptime end_fitting = boost::posix_time::microsec_clock::local_time();
        boost::posix_time::time_duration diff_detect = end_detect - start_detect;
        boost::posix_time::time_duration diff_match = end_match - start_match;
        boost::posix_time::time_duration diff_fit = end_fitting - start_fitting;

        string string_time_detect = to_string(diff_detect.total_milliseconds());
        string string_time_match = to_string(diff_match.total_milliseconds());
        string string_time_fitting = to_string(diff_fit.total_milliseconds());
        string result = to_string(detected_classes);
        string all_classes = to_string((int) target_images.size());

        rectangle(input_image, Point2d(input_image.cols - 160, 8), Point2d(input_image.cols, 22), CV_RGB(128, 128, 128),
                  CV_FILLED);
        rectangle(input_image, Point2d(input_image.cols - 160, 28), Point2d(input_image.cols, 42),
                  CV_RGB(128, 128, 128), CV_FILLED);
        rectangle(input_image, Point2d(input_image.cols - 160, 48), Point2d(input_image.cols, 62),
                  CV_RGB(128, 128, 128), CV_FILLED);
        rectangle(input_image, Point2d(input_image.cols - 160, 68), Point2d(input_image.cols, 82),
                  CV_RGB(128, 128, 128), CV_FILLED);

        putText(input_image, "Detection: " + string_time_detect + " ms", Point2d(input_image.cols - 160, 20), fontFace,
                fontScale, Scalar(255, 255, 255), 1);
        putText(input_image, "Matching: " + string_time_match + " ms", Point2d(input_image.cols - 160, 40), fontFace,
                fontScale, Scalar(255, 255, 255), 1);
        putText(input_image, "Fitting: " + string_time_fitting + " ms", Point2d(input_image.cols - 160, 60), fontFace,
                fontScale, Scalar(255, 255, 255), 1);
        putText(input_image, "Found: " + result + " of " + all_classes, Point2d(input_image.cols - 160, 80), fontFace,
                fontScale, Scalar(255, 255, 255), 1);
        // END fitting //////////////////////////////////////////////////////////////////

        mtx.unlock();
    }
}