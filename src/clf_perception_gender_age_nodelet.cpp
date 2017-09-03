// STD
#include <stdlib.h>

// ROS NODELETS
#include <ros/ros.h>
#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>

// MSGS
#include <clf_perception_vision/ExtenedPeople.h>
#include <clf_perception_vision/ExtendedPersonStamped.h>

// SELF
#include "ros_grabber.hpp"
#include "dlib_detection.hpp"

using namespace std;
using namespace cv;

namespace clf_perception_vision {

    class GenderAge : public nodelet::Nodelet {
    public:
        GenderAge() {}

    private:
        ros::NodeHandle private_nh;
        ros::Subscriber pose_sub;
        ros::Publisher people_pub;

        ROSGrabber *ros_grabber;
        DlibFace dlf;

        string cfg_yaml,
               topic,
               out_topic,
               people_topic,
               shape_mode_path,
               model_file_age,
               model_file_gender,
               trained_file_age,
               trained_file_gender,
               mean_file,
               label_file_age,
               label_file_gender;

        unsigned int _pyr = 0;
        unsigned int gender_age = 1;
        unsigned int frame_count = 0;
        unsigned int average_frames = 0;
        unsigned int microseconds = 1000;
        const int fontFace = cv::FONT_HERSHEY_PLAIN;

        double time_spend = 0;
        const double fontScale = 1;

        virtual void onInit() {
            // Get the NodeHandle
            private_nh = getPrivateNodeHandle();
            // How many CPUs do we have?
            cout << ">>> Found --> " << cv::getNumberOfCPUs() << " CPUs"<< endl;
            // Are we using optimized OpenCV Code?
            cout << ">>> OpenCV was built with optimizations --> " << cv::useOptimized() << endl;

            if (private_nh.getParam("gender_age_config", cfg_yaml))
            {
                ROS_INFO(">>> Config File: --> %s", cfg_yaml.c_str());
            } else {
                ROS_ERROR(">>> Failed to get depth topic parameter");
                exit(EXIT_FAILURE);
            }

            if (private_nh.getParam("gender_age_in_topic", people_topic))
            {
                ROS_INFO(">>> Input People Topic: --> %s", people_topic.c_str());
            } else {
                ROS_ERROR(">>> Failed to get people topic parameter");
                exit(EXIT_FAILURE);
            }

            if (private_nh.getParam("gender_age_out_topic", out_topic))
            {
                ROS_INFO(">>> Output People Topic: --> %s", out_topic.c_str());
            } else {
                ROS_ERROR(">>> Failed to get out topic parameter");
                exit(EXIT_FAILURE);
            }

            if (private_nh.getParam("gender_age_image_topic", topic))
            {
                ROS_INFO(">>> Input Image Topic: --> %s", topic.c_str());
            } else {
                ROS_ERROR(">>> Failed to get input image topic parameter");
                exit(EXIT_FAILURE);
            }

            FileStorage fs(cfg_yaml, FileStorage::READ);

            if (fs.isOpened()) {
                fs["dlib_shapepredictor"] >> shape_mode_path;
                cout << ">>> Frontal Face: --> " << shape_mode_path << endl;

                fs["pyr_up"] >> (int)_pyr;
                if (_pyr > 0) {
                    cout << ">>> Image Scaling is: --> ON" << endl;
                } else {
                    cout << ">>> Image Scaling is: --> OFF" << endl;
                }

                fs["model_file_gender"] >> model_file_gender;
                cout << ">>> Caffee Model Gender: --> " << model_file_gender << endl;

                fs["model_file_age"] >> model_file_age;
                cout << ">>> Caffee Model Age: --> " << model_file_age << endl;

                fs["trained_file_gender"] >> trained_file_gender;
                cout << ">>> Caffee Trained Gender: --> " << trained_file_gender << endl;

                fs["trained_file_age"] >> trained_file_age;
                cout << ">>> Caffee Trained Age: --> " << trained_file_age << endl;

                fs["mean_file"] >> mean_file;
                cout << ">>> Caffe Mean: --> " << mean_file << endl;

                fs["label_file_gender"] >> label_file_gender;
                cout << ">>> Labels Gender: --> " << label_file_gender << endl;

                fs["label_file_age"] >> label_file_age;
                cout << ">>> Labels Age: --> " << label_file_age << endl;
            }

            fs.release();

            cout << ">>> ROS In Topic: --> " << topic << endl;

            pose_sub = private_nh.subscribe(people_topic.c_str(), 1, &GenderAge::person_callback, this);
            people_pub = private_nh.advertise<clf_perception_vision::ExtenedPeople>(out_topic.c_str(), 1);

            ros_grabber = new ROSGrabber(topic);
            ros_grabber->setPyr(_pyr);

            dlf.setup(shape_mode_path);
            dlf.cl = new Classifier(model_file_gender, trained_file_gender, mean_file, label_file_gender);
            dlf.cl_age = new Classifier(model_file_age, trained_file_age, mean_file, label_file_age);

            cout << ">>> Ready. Let's go..." << endl;
        }

        cv::Rect dlib2cvrect(const dlib::rectangle& r) {return cv::Rect(r.left(), r.top(), r.width(), r.height());}

        void person_callback(const clf_perception_vision::ExtenedPeople::ConstPtr &person) {
            cout << "Callback gender age" << endl;
        }

    };

    PLUGINLIB_DECLARE_CLASS(clf_perception_vision, GenderAge, clf_perception_vision::GenderAge, nodelet::Nodelet
    );
}
