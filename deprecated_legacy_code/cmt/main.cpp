// SELF
#include "CMT.h"
#include "gui.h"
#include "ros_grabber.hpp"

// CV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/utility.hpp>

// STD
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdio>
// MSGS
#include "clf_perception_vision/CMTObjectTrack.h"
#include "clf_perception_vision/CMTStopObjectTrack.h"
#include <sensor_msgs/image_encodings.h>

#ifdef __GNUC__
#include <getopt.h>
#else

#include "getopt/getopt.h"

#endif

/*
 * TODO List:
 * - Bug: segfault bei mehrmaligem neustarten des trackens
 *      - tritt nur (?) bei sehr wenigen (< 10) punkten auf
 *      - und in der process_frame Methode des CMT, bzw in Methoden der Consensus Klasse
 *      - Grund: CMT/Consensus scheint neuinitialisieren nicht zu mögen
 *      - Lösung: für jedes tracking einen neuen CMT erstellen
 * - state designen
 * - state generieren
 * - state publishen
 * - object Erkennung (low prio)
 */

using cmt::CMT;
using cv::imread;
using cv::namedWindow;
using cv::Scalar;
using cv::VideoCapture;
using cv::waitKey;
using std::cerr;
using std::istream;
using std::ifstream;
using std::stringstream;
using std::ofstream;
using std::cout;
using std::min_element;
using std::max_element;
using std::endl;
using ::atof;

static string WIN_NAME = "CMT";

string topic = "/usb_cam/image_raw";
bool pyr = false;
float UPPER_I = 0;
//Create a CMT object
cmt::CMT cmt_;
int tracker_counter = 0;
unsigned int last_computed_frame = -1;
cv::Rect rect;


int display(Mat im, CMT &cmt, float result) {
    if (result > UPPER_I) {
        UPPER_I = result;
    }
    float fraction = result / UPPER_I;
    //cout << result << ", " << fraction << endl;
    if (fraction > 0.75) {
        for (size_t i = 0; i < cmt.points_active.size(); i++) {
            circle(im, cmt.points_active[i], 2, Scalar(0, 255, 0));
        }

        Point2f vertices[4];
        cmt.bb_rot.points(vertices);
        for (int i = 0; i < 4; i++) {
            line(im, vertices[i], vertices[(i + 1) % 4], Scalar(0, 255, 0));
        }
    }

    imshow(WIN_NAME, im);
    return waitKey(5);
}

bool stop_track(clf_perception_vision::CMTStopObjectTrack::Request &req,
                         clf_perception_vision::CMTStopObjectTrack::Response &res) {
    tracker_counter = 0;
    last_computed_frame = -1;
    res.success = true;
    return true;
}

bool track(clf_perception_vision::CMTObjectTrack::Request &req,
           clf_perception_vision::CMTObjectTrack::Response &res) {
    //Initialize CMT
    rect.x = req.xmin;
    rect.y = req.ymin;
    rect.width = req.xmax - req.xmin;
    rect.height = req.ymax - req.ymin;
    cout << rect.x << " " << rect.y << " " << rect.width << " " << rect.height << std::endl;
    res.success = true;

    tracker_counter = 1;
    return true;
}

int main(int argc, char **argv) {
    // ROS
    // FILELog::ReportingLevel() = logDEBUG;
    FILELog::ReportingLevel() = logINFO;

    ros::init(argc, argv, "clf_cmt", ros::init_options::AnonymousName);


    //Parse args
    int challenge_flag = 0;
    int loop_flag = 0;

    const int detector_cmd = 1000;
    const int descriptor_cmd = 1001;
    const int bbox_cmd = 1002;
    const int no_scale_cmd = 1003;
    const int with_rotation_cmd = 1004;
    const int skip_cmd = 1005;
    const int skip_msecs_cmd = 1006;
    const int output_file_cmd = 1007;

    //Create window
    namedWindow(WIN_NAME);

    bool show_preview = true;

    cv::CommandLineParser parser(argc, argv, "{@config |<none>| yaml config file}" "{help h ||}");
    cv::FileStorage fs(argv[1], cv::FileStorage::READ);

    if (fs.isOpened()) {

        fs["input_ros_topic"] >> topic;
        cout << ">>> Input Topic: --> " << topic << endl;

        fs["pyr"] >> pyr;
        cout << ">>> PyrUP: --> " << pyr << endl;
    }

    fs.release();

    // ROS
    ROSGrabber ros_grabber(topic);
    ros_grabber.setPyr(pyr);

    ros::ServiceServer track_ob(ros_grabber.node_handle_.advertiseService("/cmt/track_object", track));
    ros::ServiceServer track_obj_stop(ros_grabber.node_handle_.advertiseService("/cmt/stop_track_object", stop_track));

    //Initialize CMT
    rect.x = 320 - 50;
    rect.y = 240 - 50;
    rect.width = 50;
    rect.height = 50;

    //The image
    Mat im;
    Mat im_gray;
    float result = 0;

    //Main loop
    while (true) {

        ros::spinOnce();
        ros_grabber.getImage(&im);
        if (tracker_counter == 0) {
            continue;
        }


        else if (tracker_counter == 1) {
            //set everything to start
            Mat leere;
            im = leere;
            im_gray = leere;
            result = 0;
            UPPER_I = 0;
            cmt::CMT cmt__;
            cmt_ = cmt__;
            last_computed_frame = -1;


            while (im.empty()) {
                ros::spinOnce();
                ros_grabber.getImage(&im);
            }

            cvtColor(im, im_gray, CV_BGR2GRAY);
            cmt_.initialize(im_gray, rect);
            UPPER_I = (float) cmt_.points_active.size();
            tracker_counter = 2;



        } else if (tracker_counter == 2) {

            int tmp_frame_nr = ros_grabber.getLastFrameNr();
            if (last_computed_frame != tmp_frame_nr) {
                if (im.empty()) continue; //Exit at end of video stream
                if (im.channels() > 1) {
                    cvtColor(im, im_gray, CV_BGR2GRAY);
                } else {
                    im_gray = im;
                }
                // Let CMT process the frame
                cmt_.processFrame(im_gray);
                result = (float) cmt_.points_active.size();
            }
            last_computed_frame = ros_grabber.getLastFrameNr();
            // Display image and then quit if requested.
            char key = display(im, cmt_, result);
            if (key == 'q') break;
        }
    }

    return 0;
}
