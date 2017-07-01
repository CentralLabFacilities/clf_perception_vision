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