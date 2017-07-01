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

#include <opencv2/core.hpp>
#include <opencv2/face.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <sstream>

using namespace cv;
using namespace cv::face;
using namespace std;

GenderDetector::GenderDetector(){}
GenderDetector::~GenderDetector(){}

void GenderDetector::setup(string saved_model) {
    model = FisherFaceRecognizer::create();
    model->read(saved_model);
}

int GenderDetector::detect(Mat input_image) {
    // To get the confidence of a prediction call the model with:
    //
    //      int predictedLabel = -1;
    //      double confidence = 0.0;
    //      model->predict(testSample, predictedLabel, confidence);
    //
    resize(input_image, resized_target, Size(64,64))

    int predictedLabel = model->predict(resized_target);

    if (predictedLabel > 0) {
        cout << ">>> Male" << endl;
    } else {
        cout << ">>> Female" << endl;
    }

    return predictedLabel;
}
