/*
 * Copyright (c) 2011. Philipp Wagner <bytefish[at]gmx[dot]de>.
 * Copyright (c) 2017. Florian Lier <fl[at]techfak.uni-bielefeld[dot]de>.
 *
 * Released to public domain under terms of the BSD Simplified license.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *   * Neither the name of the organization nor the names of its contributors
 *     may be used to endorse or promote products derived from this software
 *     without specific prior written permission.
 *
 *   See <http://www.opensource.org/licenses/bsd-license>
 */


#include "clf_2d_gender.hpp"

using namespace cv;
using namespace cv::face;
using namespace std;

GenderDetector::GenderDetector(){}
GenderDetector::~GenderDetector(){}

int predictedLabel = -1;
double confidence = 0.0;
int result = -1;

void GenderDetector::setup(string saved_model) {
    model = FisherFaceRecognizer::create();
    model->read(saved_model);
}

int GenderDetector::detect(Mat input_image) {

    // Traing data is based on 100x100 patches
    resize(input_image, resized_target, Size(80,80));

    if (!resized_target.empty()) {
        cvtColor(resized_target, resized_grey_target , COLOR_BGR2GRAY);
        model->predict(resized_grey_target, predictedLabel, confidence);
        if (confidence > 42) {
            result = predictedLabel;
        } else {
            result = -1;
        }
    }

    return result;
}
