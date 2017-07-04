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

// RSC
#include <rsc/misc/langutils.h>
#include <rsc/threading/SynchronizedQueue.h>

// RSB
#include <rsb/util/EventQueuePushHandler.h>
#include <rsb/MetaData.h>
#include <rsb/Listener.h>
#include <rsb/Factory.h>
#include <rsb/filter/ScopeFilter.h>
#include <rsb/converter/Converter.h>
#include <rsb/converter/Repository.h>
#include <rsb/converter/ProtocolBufferConverter.h>
#include <rsb/Factory.h>
#include <rsb/EventCollections.h>
#include <rsb/converter/EventsByScopeMapConverter.h>

// RST
#include <rst/math/Vec2DInt.pb.h>
#include <rst/converters/opencv/IplImageConverter.h>

// STD
#include <string>
#include <iostream>
#include <sstream>
#include <mutex>
#include <stdlib.h>

// CV
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

// BOOST
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/format.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/lexical_cast.hpp>

class RSBGrabber {

public:
    RSBGrabber(std::string scope);
    ~RSBGrabber();
    void getImage(cv::Mat *image);
    std::string getDuration();
    boost::posix_time::ptime getTimestamp();
private:
    typedef rsc::threading::SynchronizedQueue<rsb::EventPtr> ImageQueue;
    typedef boost::shared_ptr<ImageQueue> ImageQueuePtr;
    typedef rsb::util::EventQueuePushHandler ImageHandler;
    typedef boost::shared_ptr<ImageHandler> ImageHandlerPtr;
    ImageQueuePtr imageQueue;
    ImageHandlerPtr imageHandler;
    rsb::ListenerPtr imageListener;
    rsb::MetaData imageMetaData;
    std::recursive_mutex mtx;
    boost::posix_time::ptime timestamp;
    std::string duration;
};