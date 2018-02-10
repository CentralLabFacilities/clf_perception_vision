#!/usr/bin/env python

# Author: Felix Friese [ffriese AT techfak.uni-bielefeld DOT de]

import time
import rospy
from optparse import OptionParser
from tf import TransformListener
from geometry_msgs.msg import Pose, PoseStamped
from clf_perception_vision_msgs.msg import ExtendedPeople, ExtendedPersonStamped


class ExtendedPeople2Map:

    def __init__(self, _in, _out):
        rospy.init_node('clf_perception_vision_people2map', anonymous=True)
        self.tf_listener = TransformListener()
        self.reference_frame = "map"
        self.sub = rospy.Subscriber(str(_in), ExtendedPeople, self.people_cb, queue_size=2)
        self.pub = rospy.Publisher(str(_out), ExtendedPeople, queue_size=2)
        rospy.loginfo(">>> People2Map is ready.")

    def people_cb(self, data):
        try:
            for p in data.persons:
                print "looking up %s --> %s " % (self.reference_frame, data.header.frame_id)
                self.tf_listener.waitForTransform(self.reference_frame, data.header.frame_id, rospy.Time.now(), rospy.Duration(1))
                trans_pose = self.tf_listener.transformPose(self.reference_frame, p.pose)
                print trans_pose
                # pa.poses.append(trans_pose)
            if len(pa.poses) > 0:
                # self.pub.publish(data)
                pass
        except Exception, ex:
            rospy.logwarn(">>> Problem looking up TF data ---> %s" % str(ex))


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

