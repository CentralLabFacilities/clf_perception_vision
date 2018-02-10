#!/usr/bin/env python

# Author: Felix Friese [ffriese AT techfak.uni-bielefeld DOT de]

import time
import rospy
from optparse import OptionParser
from tf import TransformListener
from geometry_msgs.msg import PoseArray, Pose
from clf_perception_vision_msgs.msg import ExtendedPeople


class ExtendedPeople2Map:

    def __init__(self, _in, _out):
        rospy.init_node('clf_perception_vision_people2map', anonymous=True)
        self.tf = TransformListener()
        self.reference_frame = "map"
        self.sub = rospy.Subscriber(str(_in), ExtendedPeople, self.people_cb, queue_size=2)
        self.pub = rospy.Publisher(str(_out), PoseArray, queue_size=2)
        rospy.loginfo(">>> People2Map is ready.")

    def people_cb(self, data):
        pa = PoseArray()
        pa.header = data.header
        try:
            for person in data.persons:
                # print person.pose
                if self.tf.waitForTransform(self.reference_frame, person.transformid, person.pose.header.stamp,
                                            rospy.Duration(0.5)):
                    print "found transform"
                    # target_frame, stamped_in, stamped_out
                    self.tf.transformPose(self.reference_frame, person.pose, person.pose)
                    pa.poses.append(person.pose)
            if len(pa.poses) > 0:
                self.pub.publish(pa)
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

