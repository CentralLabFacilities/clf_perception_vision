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
#include <fstream>
#include <sstream>

using namespace cv;
using namespace cv::face;
using namespace std;

static Mat norm_0_255(InputArray _src) {
    Mat src = _src.getMat();
    Mat dst;
    switch(src.channels()) {
    case 1:
        cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
        break;
    case 3:
        cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
        break;
    default:
        src.copyTo(dst);
        break;
    }
    return dst;
}

static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
    std::ifstream file(filename.c_str(), ifstream::in);
    if (!file) {
        string error_message = ">>> No valid input file was given, please check the given filename.";
        CV_Error(Error::StsBadArg, error_message);
    }
    string line, path, classlabel;
    while (getline(file, line)) {
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        if(!path.empty() && !classlabel.empty()) {
            images.push_back(imread(path, 0));
            labels.push_back(atoi(classlabel.c_str()));
        }
    }
}

int main(int argc, const char *argv[]) {
    if (argc < 2) {
        cout << ">>> usage: " << argv[0] << " <input.csv> <output_folder> " << endl;
        exit(1);
    }
    string output_folder = ".";
    if (argc == 3) {
        output_folder = string(argv[2]);
    }

    std::string prefix(argv[2]);
    string fn_csv = string(argv[1]);
    vector<Mat> images;
    vector<int> labels;
    try {
        read_csv(fn_csv, images, labels);
    } catch (cv::Exception& e) {
        cerr << ">>> Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
        exit(1);
    }
    if(images.size() <= 1) {
        string error_message = ">>> This training needs at least 2 images to work. Please add more images to your data set!";
        CV_Error(Error::StsError, error_message);
    }
    int height = images[0].rows;
    Mat testSample = images[images.size() - 1];
    int testLabel = labels[labels.size() - 1];
    images.pop_back();
    labels.pop_back();
    Ptr<FisherFaceRecognizer> model = FisherFaceRecognizer::create();
    model->train(images, labels);
    model->write(prefix+"/fisher_gender.xml");
    cout << ">>> Model saved to: " << prefix+"/fisher_gender.xml" << endl;
    int predictedLabel = -1;
    double confidence = 0.0;
    int result = -1;
    model->predict(testSample, predictedLabel, confidence);
    result = predictedLabel;
    string result_message = format(">>> Predicted class = %d / Actual class = %d / Confidence = %f", result, testLabel, confidence);
    cout << result_message << endl;
    Mat eigenvalues = model->getEigenValues();
    Mat W = model->getEigenVectors();
    Mat mean = model->getMean();
    return 0;
}
