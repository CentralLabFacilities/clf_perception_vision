#!/usr/bin/env python

# Author: Felix Friese [ffriese AT techfak.uni-bielefeld DOT de]

import time
import rospy
from optparse import OptionParser
from tf import TransformListener
from clf_perception_vision_msgs.msg import ExtendedPeople


class ExtendedPeople2Map:

    def __init__(self, _in, _out):
        rospy.init_node('clf_perception_vision_people2map', anonymous=True)
        self.tf = TransformListener()
        self.reference_frame = "map"
        self.sub = rospy.Subscriber(str(_in), ExtendedPeople, self.people_cb, queue_size=2)
        self.pub = rospy.Publisher(str(_out), ExtendedPeople, queue_size=2)
        rospy.loginfo(">>> People2Map is ready.")

    def people_cb(self, data):

        frame_id = data.persons[0].pose.header.frame_id
        if not self.tf.frameExists(frame_id):
            rospy.logwarn(">>> Frame does not exist, %s" % frame_id)
            time.sleep(0.05)
            return
        if not self.tf.frameExists(self.reference_frame):
            rospy.logwarn(">>> Frame does not exist, %s" % self.reference_frame)
            time.sleep(0.05)
            return
        try:
            for person in data.persons:
                self.tf.waitForTransform(self.reference_frame, person.pose, rospy.Time.now(), rospy.Duration(0.15))
                self.tf.transformPose(self.reference_frame, person.pose)
            self.pub.publish(data)
        except Exception, ex:
            rospy.logwarn(">>> Problem receiving data, %s" % str(ex))


if __name__ == "__main__":

    parser = OptionParser()
    parser.add_option("--intopic", dest="intopic", default="/clf_perception_vision/people/raw/transform")
    parser.add_option("--outtopic", dest="outtopic", default="/clf_perception_vision/people/raw/transform/map")
    (options, args) = parser.parse_args()
    P2M = ExtendedPeople2Map(options.intopic, options.outtopic)
    while not rospy.is_shutdown():
        try:
            rospy.spin()
        except rospy.ROSInterruptException, ex:
            rospy.logwarn(">>> Exiting, %s" % str(ex))

