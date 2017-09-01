#!/usr/bin/env python

import rospy
import std_msgs.msg
from darknet_ros_msgs.msg import BoundingBoxes, BoundingBox
from clf_perception_vision.msg import ExtenedPeople, ExtendedPersonStamped


def box_cb(data):
    try:
        e = ExtenedPeople()
        h = std_msgs.msg.Header()
        h.stamp = rospy.Time.now()
        for person in data.boundingBoxes:
            if person.Class == "person":
                p = ExtendedPersonStamped()
                p.header = h
                p.bbox_xmin = person.xmin
                p.bbox_xmax = person.xmax
                p.bbox_ymin = person.ymin
                p.bbox_ymax = person.ymax
                p.probability = person.probability
                e.persons.append(p)

    except Exception, e:
        rospy.logwarn(">>> problem receiving data, %s" % str(e))


rospy.init_node('clf_perception_vision_box2people', anonymous=True)
rate = rospy.Rate(2.0)
sub = rospy.Subscriber('/darknet_ros/bounding_boxes', BoundingBoxes, box_cb)
pub = rospy.Publisher('/clf_perception_vision_box2people/people_raw', ExtenedPeople, queue_size=1)


def publish_extended_people(data):
    pass


if __name__ == '__main__':
    try:
        while not rospy.is_shutdown():
            rate.sleep()
    except rospy.ROSInterruptException:
        pass
