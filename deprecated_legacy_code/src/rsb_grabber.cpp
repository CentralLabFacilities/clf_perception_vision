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
#include "rsb_grabber.hpp"

using namespace std;
using namespace rsc::logging;
using namespace rst::vision;
using namespace rst::math;


RSBGrabber::RSBGrabber(std::string _scope) {

    imageQueue = ImageQueuePtr(new ImageQueue(1));
    imageHandler = ImageHandlerPtr(new ImageHandler(imageQueue));

    rsb::Factory &factory = rsb::getFactory();

    try {
        rsb::converter::Converter<std::string>::Ptr image_c(new rst::converters::opencv::IplImageConverter());
        rsb::converter::converterRepository<string>()->registerConverter(image_c);
    } catch(...) {
        cout << ">> RSB IS WEIRD (converter already registered)" << endl;
    }

    imageListener = factory.createListener(rsb::Scope(_scope));
    imageListener->addHandler(imageHandler);
}

RSBGrabber::~RSBGrabber() {}

void RSBGrabber::getImage(cv::Mat *image) {
    boost::posix_time::ptime init = boost::posix_time::microsec_clock::local_time();
    rsb::EventPtr imageEvent = imageQueue->pop();
    timestamp = boost::posix_time::microsec_clock::local_time();
    mtx.lock();
    if (imageEvent->getType() == rsc::runtime::typeName<IplImage>()) {
        boost::shared_ptr<IplImage> newImage = boost::static_pointer_cast<IplImage>(imageEvent->getData());
        imageMetaData = imageEvent->getMetaData();
        (*image) = cv::Mat(newImage.get(), true);
    }
    mtx.unlock();

    boost::posix_time::ptime c = boost::posix_time::microsec_clock::local_time();
    boost::posix_time::time_duration cdiff = c - init;
    duration = std::to_string(cdiff.total_milliseconds());
}

std::string RSBGrabber::getDuration() {
    return duration;
}

boost::posix_time::ptime RSBGrabber::getTimestamp() {
    return timestamp;
}
