#!/usr/bin/env python

import rospy
import std_msgs.msg
from optparse import OptionParser
from darknet_ros_msgs.msg import BoundingBoxes, BoundingBox
from clf_perception_vision.msg import ExtendedPeople, ExtendedPersonStamped


class BBox2ExtendedPeople:

    def __init__(self, _in, _out):
        rospy.init_node('clf_perception_vision_box2people', anonymous=True)
        self.sub = rospy.Subscriber(str(_in), BoundingBoxes, self.box_cb, queue_size=2)
        self.pub = rospy.Publisher(str(_out), ExtendedPeople, queue_size=2)
        rospy.logdebug(">>> Box2People is ready.")

    def box_cb(self, data):
        try:
            e = ExtendedPeople()
            h = std_msgs.msg.Header()
            h.stamp = rospy.Time.now()
            e.header = h
            for person in data.boundingBoxes:
                if person.Class == "person" and float(person.probability) > 0.31:
                    p = ExtendedPersonStamped()
                    p.header = h
                    p.bbox_xmin = person.xmin
                    p.bbox_xmax = person.xmax
                    p.bbox_ymin = person.ymin
                    p.bbox_ymax = person.ymax
                    p.probability = float('%.2f' % person.probability)
                    e.persons.append(p)
            self.pub.publish(e)
        except Exception, ex:
            rospy.logwarn(">>> Problem receiving data, %s" % str(ex))


if __name__ == "__main__":

    parser = OptionParser()
    parser.add_option("--intopic", dest="intopic", default="/darknet_ros/bounding_boxes")
    parser.add_option("--outtopic", dest="outtopic", default="/clf_perception_vision/people/raw")
    (options, args) = parser.parse_args()
    B2P = BBox2ExtendedPeople(options.intopic, options.outtopic)
    rate = rospy.Rate(2.0)
    while not rospy.is_shutdown():
        try:
            rospy.spin()
        except rospy.ROSInterruptException, ex:
            rospy.logwarn(">>> Exiting, %s" % str(ex))

