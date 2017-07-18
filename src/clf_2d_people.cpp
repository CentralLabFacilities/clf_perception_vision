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
#include "clf_2d_people.hpp"


PeopleDetector::PeopleDetector() { }
PeopleDetector::~PeopleDetector() { }

void PeopleDetector::setup() {
    win_stride_width = 8;
    win_stride_height = 8;
    win_width = 48;
    block_width = 16;
    block_stride_width = 8;
    block_stride_height = 8;
    cell_width = 8;
    nbins = 9;

    scale = 1.05;
    nlevels = 13;
    gr_threshold = 8;
    hit_threshold = 1.4;
    hit_threshold_auto = true;

    cv::Size win_stride(win_stride_width, win_stride_height);
    cv::Size win_size(win_width, win_width * 2);
    cv::Size block_size(block_width, block_width);
    cv::Size block_stride(block_stride_width, block_stride_height);
    cv::Size cell_size(cell_width, cell_width);

    cuda_hog = cv::cuda::HOG::create(win_size, block_size, block_stride, cell_size, nbins);
    people_detector = cuda_hog->getDefaultPeopleDetector();
    cuda_hog->setSVMDetector(people_detector);

    std::cout << "CUDA HOG Descriptor Size: --> " << cuda_hog->getDescriptorSize() << std::endl;

    cuda_hog->setNumLevels(nlevels);
    cuda_hog->setHitThreshold(hit_threshold);
    cuda_hog->setWinStride(win_stride);
    cuda_hog->setScaleFactor(scale);
    cuda_hog->setGroupThreshold(gr_threshold);
}

std::vector<cv::Rect> PeopleDetector::detect(cv::Mat img) {
    people_cuda_img.upload(img);
    cv::cuda::cvtColor(people_cuda_img, people_cuda_img_grey, cv::COLOR_BGR2GRAY);
    cuda_hog->detectMultiScale(people_cuda_img_grey, people_found);
    return people_found;
}