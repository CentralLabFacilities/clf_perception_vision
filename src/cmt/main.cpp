// SELF
#include "CMT.h"
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
#include <unistd.h>

// MSGS
#include <clf_perception_vision_msgs/CMTrackerResults.h>
#include <clf_perception_vision_msgs/CMTObjectTrack.h>
#include <clf_perception_vision_msgs/CMTStopObjectTrack.h>
#include <sensor_msgs/image_encodings.h>

#ifdef __GNUC__
#include <getopt.h>
#else

#include "getopt/getopt.h"

#endif

/*
 * TODO List:
 * - main loop frisst massiv cpu
 *      - liegt vmtl am continue bei tracker_counter = 0
 *      - mögliche lösung: sleep in abhängigkeit von fps (config)?
 *      - Done
 * - state designen
 *      - angle | float
 *      - tracked | bool
 *      - position | 2 int
 * - state generieren + publishen
 * - streamende: was tun?
 *      - tracking des objects beenden
 *      - warten, bis es wieder bilddateien gibt (aktuelle lösung)
 *      - node terminieren
 * - reaquirieren von halb verlorenen objekten
 *      - wenn alle features verloren sind, prüft der cmt auf dem ganzen bild nach den punkten
 *          - verhalten vmtl nur nebeneffekt, wenn im vorherigen frame nichts gefunden wurde
 *          - berichtigung: es wird immer das gesamte bild durchsucht. der tracker versucht die features des vergangenen
 *              frames wiederzufinden (optical flow). Diese werden dann mit den detecteten (detector mit FAST) features des aktuellen
 *              frames gematcht (matcher).
 *      - wenn nur noch wenige features gefunden werden oder der getrackte bereich zu hart springen sollte derselbe vorgang ablaufen
 *          - Done
 * - Bug: segfault bei mehrmaligem neustarten des trackens
 *      - tritt nur (?) bei sehr wenigen (< 10) punkten auf
 *      - und in der process_frame Methode des CMT, bzw in Methoden der Consensus Klasse
 *      - Grund: CMT/Consensus scheint neuinitialisieren nicht zu mögen
 *      - Lösung: für jedes tracking einen neuen CMT erstellen
 *      - Done
 * - Nur jedes x-te bild zum tracken nutzen
 *      - über sleep und vergange zeit seit dem letzten bild realisieren
 * - nachdem die continuity getriggert hat, x mal das überprüfen aussetzen (weil neuaquirieren des objects iaR zu einem
 *      Sprung aka schlechter Continuity führt)
 *      - Done
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

string topic = "/webcam/image_raw";
string state_topic = "/cmt/tracking_results";
bool pyr = false;
bool show_tracking_results = false;
float UPPER_I = 0;
//Create a CMT object
cmt::CMT cmt_;
int sensor_fps = 5;
unsigned int last_computed_frame = -1;
cv::Rect rect;
enum Tracking_Status {
    idle, start, running
};
Tracking_Status tracking_status = idle;
float tracking_lost_counter = 0.0;
ros::Publisher tracker_pub_;

/*
 * Config File needed.
 * In it declared should be (with example/ default values):
 * sensor_fps: 30
 * pyr: 0
 * input_ros_topic: "/usb_cam/image_raw"
 * output_ros_topic: "/cmt/state"
 * show_tracking_results: 1
 */

/**
 * Will display the momentary webcam image with the outline of the tracked rectangle and the found features.
 * @param im the picture as openCV mat
 * @param cmt cmt which tracks the rectangle
 * @param result what button was pressed while the image was displayed (for escaping etc)
 * @return
 */
int display(Mat im, CMT &cmt, float result) {
    Mat im_cp = Mat(im);
    float fraction = result / UPPER_I;
    cout << result << ", " << fraction << endl;
    for (size_t i = 0; i < cmt.points_active.size(); i++) {
        circle(im_cp, cmt.points_active[i], 2, Scalar(0, 255, 0));
    }

    Point2f vertices[4];
    cmt.bb_rot.points(vertices);
    for (int i = 0; i < 4; i++) {
        line(im_cp, vertices[i], vertices[(i + 1) % 4], Scalar(0, 255, 0));
    }

    imshow(WIN_NAME, im_cp);
    return waitKey(5);
}

/**
 * Callback method for the rosservice which stops the tracking. Request message is mostly irrelevant. Will stop the
 * tracking (and therefore publishing of tracking positions)
 * @param req the request msg (content irrelevant)
 * @param res
 * @return
 */
bool stop_track(clf_perception_vision_msgs::CMTStopObjectTrack::Request &req,
                clf_perception_vision_msgs::CMTStopObjectTrack::Response &res) {
    tracking_status = idle;
    last_computed_frame = -1;
    res.success = true;
    return true;
}

/**
 * Callback for the rosservice which starts the tracking. Request message should contain the x and y min and max values
 * of the initial rectangle.
 * @param req
 * @param res
 * @return
 */
bool track(clf_perception_vision_msgs::CMTObjectTrack::Request &req,
           clf_perception_vision_msgs::CMTObjectTrack::Response &res) {
    //Initialize CMT
    rect.x = req.xmin;
    rect.y = req.ymin;
    rect.width = req.xmax - req.xmin;
    rect.height = req.ymax - req.ymin;
    ROS_INFO("Starting tracking of object in region: x: %d+%d, y: %d+%d", rect.x, rect.width, rect.y, rect.height);
    res.success = true;
    tracking_lost_counter = 0;

    tracking_status = start;
    return true;
}

/**
 * Publishes the state of the tracking. Isn't really more than a wrapper that stuffs its parameters into a msg and
 * rospublishes it.
 * @param tracked whether or not the object is considered tracked. If this is false, do not trust the other parts of the
 * message.
 * @param angle the angle the object leans to. Negative values indicate left (counterclockwise) and positive values vice
 * versa
 * @param position the position where the tracked rectangle is momentarily
 */
void pub_state(bool tracked, float &angle, Point2f &position) {
    clf_perception_vision_msgs::CMTrackerResults msg_;
    msg_.tracked = tracked;
    msg_.angle = angle;
    msg_.pos_x = position.x;
    msg_.pos_y = position.y;
    tracker_pub_.publish(msg_);
}

/**
 * Will determine whether or not the object/ rectangle is momentarily tracked. It will do so by keeping track how often
 * the cmt.continuity determined the object lost (and by how many features were found at all).
 * @return True if the object is still tracked, False otherwise.
 */
bool calculate_still_tracked() {
    if (!cmt_.continuity_preserved) {
        tracking_lost_counter += 1.2 * sensor_fps;
    }
    // keep counter in range (0, 2*sensor_fps)
    if (tracking_lost_counter > 0) {
        tracking_lost_counter--;
        if (tracking_lost_counter > 2 * sensor_fps) {
            tracking_lost_counter = 2 * sensor_fps;
        }
    }
    //
    return tracking_lost_counter < 1.2 * sensor_fps && (cmt_.points_active.size() >= 5);
}

/**
 * Typical main method of the cmt tracker. Will first read the config file and initialize ros and a few variables
 * (and e.g. the cmt tracker. Then it will loop indefinitely grabbing the newest image (via rosgrabber) and, depending
 * on which state it is in (idle, starting or running), it will wait, initialize the cmt, or feed the cmt the newest
 * image (respectively)
 * @param argc
 * @param argv
 * @return
 */
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


    cv::CommandLineParser parser(argc, argv, "{@config |<none>| yaml config file}" "{help h ||}");
    cout << ">>> Config File: --> " << argv[1] << endl;
    cv::FileStorage fs(argv[1], cv::FileStorage::READ);

    if (fs.isOpened()) {

        fs["input_ros_topic"] >> topic;
        cout << ">>> Input Topic: --> " << topic << endl;

        fs["output_ros_topic"] >> state_topic;
        cout << ">>> Output Topic: --> " << state_topic << endl;

        fs["pyr"] >> pyr;
        cout << ">>> PyrUP: --> " << pyr << endl;

        fs["show_tracking_results"] >> show_tracking_results;
        cout << ">>> Show Tracking Results: --> " << show_tracking_results << endl;

        fs["sensor_fps"] >> sensor_fps;
        cout << ">>> Sensor FPS: --> " << sensor_fps << endl;
    } else {
        cout << ">>> Could not open Config file." << endl;
    }

    fs.release();

    if (show_tracking_results) {
        //Create window
        namedWindow(WIN_NAME);
    }

    // ROS
    ROSGrabber ros_grabber(topic);
    ros_grabber.setPyr(pyr);

    ros::ServiceServer track_ob(ros_grabber.node_handle_.advertiseService("/cmt/track_object", track));
    ros::ServiceServer track_obj_stop(ros_grabber.node_handle_.advertiseService("/cmt/stop_track_object", stop_track));
    tracker_pub_ = ros_grabber.node_handle_.advertise<clf_perception_vision_msgs::CMTrackerResults>(state_topic,
                                                                                               1000);

    //The image
    Mat im;
    Mat im_gray;
    float result = 0;

    //Main loop
    while (true) {

        ros::spinOnce();
        ros_grabber.getImage(&im);
        switch (tracking_status) {
            case idle: {
                usleep((int) (1000 / sensor_fps));
                continue;
            }
            case start: {
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
                tracking_status = running;
                break;

            }
            case running: {
                int tmp_frame_nr = ros_grabber.getLastFrameNr();
                if (last_computed_frame != tmp_frame_nr) {
                    if (im.empty()) {
                        tracking_status = idle;
                        continue; //Exit at end of video stream
                    }
                    if (im.channels() > 1) {
                        cvtColor(im, im_gray, CV_BGR2GRAY);
                    } else {
                        im_gray = im;
                    }
                    // Let CMT process the frame
                    cmt_.processFrame(im_gray);
                    result = (float) cmt_.points_active.size();
                } else {
                    //TODO sleep for less cpu usage (how long? 1/10 frame duration?)
                    usleep((int) (1000 / (1 * sensor_fps)));
                    continue;
                }
                last_computed_frame = ros_grabber.getLastFrameNr();
                // Display image and then quit if requested.
                if (show_tracking_results) {
                    char key = display(im, cmt_, result);
                    if (key == 'q') exit(0);
                }

                //publish result
                pub_state(calculate_still_tracked(), cmt_.bb_rot.angle, cmt_.bb_rot.center);
            }
        }
    }
}
