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
#include <stdlib.h>

// CUDA
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>

// SELF
#include "clf_2d_detect.hpp"

using namespace std;
using namespace cv;

Detect2D::Detect2D(){}
Detect2D::~Detect2D(){}

vector<Scalar> Detect2D::color_mix() {
    vector<Scalar> colormix;
    // Blue
    colormix.push_back(Scalar(219, 152, 52));
    // Cyan
    colormix.push_back(Scalar(173, 68, 142));
    // Orange
    colormix.push_back(Scalar(34, 126, 230));
    // Turquoise
    colormix.push_back(Scalar(156, 188, 26));
    // Pomgranate
    colormix.push_back(Scalar(43, 57, 192));
    // Asbestos
    colormix.push_back(Scalar(141, 140, 127));
    // Emerald
    colormix.push_back(Scalar(113, 204, 46));
    // White
    colormix.push_back(Scalar(241, 240, 236));
    // Green Sea
    colormix.push_back(Scalar(133, 160, 22));
    // Black
    colormix.push_back(Scalar(0, 0, 0));

    return colormix;
}

int Detect2D::get_x_resolution() {
    return res_x;
}

int Detect2D::get_y_resolution() {
    return res_y;
}

bool Detect2D::get_silent() {
    return toggle_silent;
}

int Detect2D::setup(int argc, char *argv[]) {

    cout << ">>> Cuda Enabled Devices --> " << cuda::getCudaEnabledDeviceCount() << endl;

    if (cuda::getCudaEnabledDeviceCount() == 0)
    {
        cout << "E >>> No cuda Enabled Devices" << endl;
        exit(EXIT_FAILURE);
    } else {
        cout << ">>> ";
        cuda::printShortCudaDeviceInfo(cuda::getDevice());
    }

    CommandLineParser parser(argc, argv, "{@config |<none>| yaml config file}" "{help h ||}");
    FileStorage fs(argv[1], FileStorage::READ);

    if (fs.isOpened()) {\
        fs["keypointalgo"] >> type_descriptor;
        cout << ">>> Keypoint Descriptor --> " << type_descriptor  << endl;

        point_matcher = "BruteForce (Hamming)";
        cout << ">>> Matching Algorithm --> " << point_matcher  << endl;

        fs["maxkeypoints"] >> max_keypoints;
        cout << ">>> Keypoints: --> " << max_keypoints  << endl;

        max_number_matching_points = fs["maxmatches"];
        cout << ">>> Maximum number of matching points --> " << max_number_matching_points << endl;

        detection_threshold = fs["detectionthreshold"];
        cout << ">>> Detection Threshold --> " << detection_threshold << endl;

        scale_factor = fs["scalefactor"];
        cout << ">>> Scalefactor --> " << scale_factor << endl;

        fs["homography"] >> draw_homography;
        cout << ">>> Draw Homography --> " << draw_homography << endl;

        if (draw_homography == "true") {
            toggle_homography = true;
        }

        fs["silent"] >> draw_image;
        cout << ">>> Silent --> " << draw_image << endl;

        if (draw_image == "true") {
            toggle_silent = true;
        }

        FileNode targets = fs["targets"];
        FileNodeIterator it = targets.begin(), it_end = targets.end();

        int idx = 0;

        for( ; it != it_end; ++it, idx++ ) {
            cout << ">>> Target " << idx << " --> ";
            cout << (String)(*it) << endl;
            target_paths.push_back((String)(*it));
        }

        if(idx > 6) {
            cout << "E >>> Sorry, only 5 targets are allowed (for now)" << endl;
            exit(EXIT_FAILURE);
        }

        FileNode labels = fs["labels"];
        FileNodeIterator it2 = labels.begin(), it_end2 = labels.end();

        int idy = 0;

        for( ; it2 != it_end2; ++it2, idy++ ) {
            cout << ">>> Label  " << idy << " --> ";
            cout << (String)(*it2) << endl;
            target_labels.push_back((String)(*it2));
        }
    }

    fs.release();

    colors = color_mix();
    target_medians = new cv::Point2d[target_paths.size()];

    if (type_descriptor.compare("ORB") == 0) {
        cuda_orb = cuda::ORB::create(max_keypoints);
    } else {
        cout << "E >>> Sorry, only ORB (for now)" << endl;
        exit(EXIT_FAILURE);
    }

    cuda_bf_matcher = cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);

    for(int i=0; i < target_paths.size(); i++) {

        Mat tmp_img = imread(target_paths[i], IMREAD_GRAYSCALE);
        cuda::GpuMat cuda_tmp_img(tmp_img);

        if (tmp_img.rows*tmp_img.cols <= 0) {
            cout << "E >>> Image " << target_paths[i] << " is empty or cannot be found" << endl;
            return -1;
        }

        target_images.push_back(tmp_img);

        vector<KeyPoint> tmp_kp;
        cuda::GpuMat tmp_cuda_dc;

        try {
            cuda_orb->detectAndComputeAsync(cuda_tmp_img, cuda::GpuMat(), tmp_kp, tmp_cuda_dc);
        }
        catch (Exception& e) {
            cout << "E >>> ORB init fail O_O" << "\n";
            return -1;
        }

        keys_current_target.push_back(tmp_kp);
        cuda_desc_current_target_image.push_back(tmp_cuda_dc);
    }

    object_pub = node_handle_.advertise<visualization_msgs::InteractiveMarkerPose>("clf_2d_detect/objects", 1);

}

void Detect2D::detect(Mat input_image, std::string capture_duration, ros::Time timestamp) {

    boost::posix_time::ptime start_detect = boost::posix_time::microsec_clock::local_time();

    cuda::GpuMat cuda_frame_tmp_img(input_image);

    if (scale_factor > 1.0) {
        cuda::resize(cuda_frame_tmp_img, cuda_frame_scaled, cv::Size(), scale_factor, scale_factor, cv::INTER_LINEAR);
        cuda::cvtColor(cuda_frame_scaled, cuda_camera_tmp_img, COLOR_BGR2GRAY);
    } else {
        cuda::cvtColor(cuda_frame_tmp_img, cuda_camera_tmp_img, COLOR_BGR2GRAY);
    }

    try {
        cuda_orb->detectAndCompute(cuda_camera_tmp_img, cuda::GpuMat(), keys_camera_image, cuda_desc_camera_image);
    }
    catch (Exception& e) {
        cout << "E >>> ORB fail O_O" << "\n";
        return -1;
    }

    boost::posix_time::ptime end_detect = boost::posix_time::microsec_clock::local_time();

    if (keys_camera_image.empty()) {
        cout << "E >>> Could not derive enough keypoints: " << endl;
        return -1;
    }

    vector<double> cum_distance;
    vector<vector<DMatch>> cum_matches;

    boost::posix_time::ptime start_match = boost::posix_time::microsec_clock::local_time();

    for(int i=0; i < target_images.size(); i++) {

        vector<DMatch> matches;

        try {
            try {

                if(!cuda_desc_current_target_image[i].empty() && !cuda_desc_camera_image.empty()) {

                    cuda_bf_matcher->match(cuda_desc_current_target_image[i], cuda_desc_camera_image, matches);

                    // Keep best matches only to have a nice drawing.
                    // We sort distance between descriptor matches
                    if (matches.size() <= max_number_matching_points) {
                        cout << "E >>> Not enough matches: " << matches.size() << endl;
                        continue;
                    }

                    Mat index;
                    int nbMatch=int(matches.size());

                    Mat tab(nbMatch, 1, CV_32F);

                    for (int i = 0; i<nbMatch; i++)
                    {
                        tab.at<float>(i, 0) = matches[i].distance;
                    }

                    sortIdx(tab, index, SORT_EVERY_COLUMN + SORT_ASCENDING);

                    vector<DMatch> bestMatches;

                    for (int i = 0; i<max_number_matching_points; i++)
                    {
                        if (matches[index.at<int>(i, 0)].distance <= detection_threshold*1.5) {
                            bestMatches.push_back(matches[index.at<int>(i, 0)]);
                        }
                    }

                    cum_matches.push_back(bestMatches);

                    vector<DMatch>::iterator it;
                    double raw_distance_sum = 0;

                    for (it = bestMatches.begin(); it != bestMatches.end(); it++) {
                        raw_distance_sum = raw_distance_sum + it->distance;
                    }

                    double mean_distance = raw_distance_sum/max_number_matching_points;
                    cum_distance.push_back(mean_distance);

                } else {
                    do_not_draw = true;
                    cout << "E >>> Descriptors are empty" << endl;
                }
            }
            catch (Exception& e) {
                cout << "E >>> Cumulative distance cannot be computed, next iteration" << endl;
                continue;
            }
        }
        catch (Exception& e) {
            cout << "E >>> Matcher is wating for input..." << endl;
            continue;
        }
    }

    boost::posix_time::ptime end_match = boost::posix_time::microsec_clock::local_time();

    text_offset_y = 20;

    if (do_not_draw == false) {
        for (int i=0; i < target_images.size(); i++) {
            try {
                vector<DMatch>::iterator it;
                vector<int> point_list_x;
                vector<int> point_list_y;

                for (it = cum_matches[i].begin(); it != cum_matches[i].end(); it++) {
                    // Point2d k_t = keys_current_target[i][it->queryIdx].pt;
                    Point2d c_t = keys_camera_image[it->trainIdx].pt;

                    point_list_x.push_back(c_t.x);
                    point_list_y.push_back(c_t.y);

                    Point2d current_point(c_t.x, c_t.y );
                    Point2d current_point_draw(c_t.x/scale_factor, c_t.y/scale_factor);

                    circle(input_image, current_point_draw, 3.0, colors[i], 1, 1 );
                }

                nth_element(point_list_x.begin(), point_list_x.begin() + point_list_x.size()/2, point_list_x.end());
                nth_element(point_list_y.begin(), point_list_y.begin() + point_list_y.size()/2, point_list_y.end());

                if (!point_list_x.empty() && !point_list_y.empty()) {
                    int median_x =  point_list_x[point_list_x.size()/2];
                    int median_y = point_list_y[point_list_y.size()/2];

                    Point2d location = Point2d(median_x, median_y);
                    Point2d draw_location = Point2d(median_x/scale_factor, median_y/scale_factor);

                    target_medians[i] = location;

                    if (cum_distance[i] <= detection_threshold) {
                        putText(input_image, target_labels[i], draw_location, fontFace, fontScale, colors[i], 2);
                    }

                    string label = target_labels[i]+": ";
                    string distance_raw = to_string(cum_distance[i]);

                    putText(input_image, label+distance_raw, Point2d(text_origin, text_offset_y), fontFace, fontScale, colors[i], 1);
                    text_offset_y = text_offset_y+15;
                }
            } catch (Exception& e) {
                cout << "E >>> Could not derive median" << endl;
                continue;
            }
        }
    }

    for (int i=0; i < target_images.size(); i++) {
        if (cum_distance[i] <= detection_threshold && toggle_homography) {
            try {
                //-- localize the object
                vector<Point2d> obj;
                vector<Point2d> scene;

                vector<DMatch>::iterator it;
                for (it = cum_matches[i].begin(); it != cum_matches[i].end(); it++) {
                    obj.push_back(keys_current_target[i][it->queryIdx].pt);
                    scene.push_back(keys_camera_image[it->trainIdx].pt);
                }

                if( !obj.empty() && !scene.empty() && cum_matches[i].size() >= 4) {

                    Mat H = findHomography(obj, scene, cv::RANSAC);

                    // Get the corners from the object to be detected
                    vector<cv::Point2d> obj_corners(4);
                    obj_corners[0] = Point(0, 0);
                    obj_corners[1] = Point(target_images[i].cols, 0);
                    obj_corners[2] = Point(target_images[i].cols, target_images[i].rows);
                    obj_corners[3] = Point(0, target_images[i].rows);

                    // vector<Point2f> scene_corners_f(4);
                    vector<Point2d> scene_corners(4);
                    vector<Point2d> scene_corners_draw(4);

                    perspectiveTransform(obj_corners, scene_corners, H);

                    // TODO: Fix the view for scaled images!
                    for (size_t i=0 ; i<scene_corners.size(); i++) {
                        // scene_corners_f.push_back(cv::Point2f((float)scene_corners[i].x, (float)scene_corners[i].y));
                        double x = scene_corners[i].x/scale_factor;
                        double y = scene_corners[i].y/scale_factor;
                        scene_corners_draw.push_back(cv::Point2d(x,y));
                    }

                    // TermCriteria termCriteria = TermCriteria(TermCriteria::MAX_ITER| TermCriteria::EPS, 20, 0.01);
                    // cornerSubPix(input_image, scene_corners_f, Size(15,15), Size(-1,-1), termCriteria);

                    //-- Draw lines between the corners (the mapped object in the scene - image_2 )
                    line(input_image, scene_corners_draw[0], scene_corners_draw[1], Scalar(113, 204, 46), 4 );
                    line(input_image, scene_corners_draw[1], scene_corners_draw[2], Scalar(113, 204, 46), 4 );
                    line(input_image, scene_corners_draw[2], scene_corners_draw[3], Scalar(113, 204, 46), 4 );
                    line(input_image, scene_corners_draw[3], scene_corners_draw[0], Scalar(113, 204, 46), 4 );

                    int diff_0 = scene_corners[1].x - scene_corners[0].x;
                    int diff_1 = scene_corners[2].y - scene_corners[1].y;

                    if (diff_0 > 0 && diff_1 > 0) {
                        int angle = int(atan((scene_corners[1].y-scene_corners[2].y)/(scene_corners[0].y-scene_corners[1].y))*180/M_PI);
                        if (abs(angle) > 80 && abs(angle) < 95) {
                            h.stamp = timestamp;
                            h.frame_id = "camera";
                            msg.header = h;
                            pt.x = target_medians[i].x;
                            pt.y = target_medians[i].y;
                            pt.z = target_medians[i].x*target_medians[i].y;
                            msg.pose.position = pt;
                            msg.name = target_labels[i];
                            object_pub.publish(msg);
                        }
                    }

                }

            } catch (Exception& e) {
                cout << "E >>> Could not derive transform" << endl;
            }
        }

        if (cum_distance[i] <= detection_threshold && !toggle_homography){
            h.stamp = timestamp;
            h.frame_id = "camera";
            msg.header = h;
            pt.x = target_medians[i].x;
            pt.y = target_medians[i].y;
            pt.z = target_medians[i].x*target_medians[i].y;
            msg.pose.position = pt;
            msg.name = target_labels[i];
            object_pub.publish(msg);
        }
    }

    boost::posix_time::time_duration diff_detect = end_detect - start_detect;
    boost::posix_time::time_duration diff_match = end_match - start_match;

    string string_time_detect = to_string(diff_detect.total_milliseconds());
    string string_time_match = to_string(diff_match.total_milliseconds());

    putText(input_image, "Delta T (Capture): "+capture_duration+" ms", Point2d(input_image.cols-280, 20), fontFace, fontScale, Scalar(156, 188, 26), 1);
    putText(input_image, "Delta T (Detect): "+string_time_detect+" ms", Point2d(input_image.cols-280, 40), fontFace, fontScale, Scalar(43, 57, 192), 1);
    putText(input_image, "Delta T (Match): "+string_time_match+" ms", Point2d(input_image.cols-280, 60), fontFace, fontScale, Scalar(34, 126, 230), 1);

}
