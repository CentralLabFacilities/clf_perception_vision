#!/usr/bin/env python

# Author: Florian Lier [flier AT techfak.uni-bielefeld DOT de]

import rospy
from optparse import OptionParser
from clf_perception_vision.msg import ExtendedPeople
from tf import TransformListener


class ExtendedPeople2Map:

    def __init__(self, _in, _out):
        rospy.init_node('clf_perception_vision_people2map', anonymous=True)
        self.tf = TransformListener()
        self.sub = rospy.Subscriber(str(_in), ExtendedPeople, self.people_cb, queue_size=2)
        self.pub = rospy.Publisher(str(_out), ExtendedPeople, queue_size=2)
        rospy.logdebug(">>> People2Map is ready.")

    def people_cb(self, data):

        frame_id = data.persons[0].pose.header.frame_id
        if not self.tf.frameExists(frame_id) and self.tf.frameExists("/map"):
            print('transform from %s to /map not found... discarding message' % frame_id)
            return

        try:
            for person in data.persons:
                person.pose = self.tf.transformPose(person.pose, '/map')

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

