#!/usr/bin/env python

import rospy
import std_msgs.msg
from optparse import OptionParser
from darknet_ros_msgs.msg import BoundingBoxes, BoundingBox
from clf_perception_vision.msg import ExtendedObjects, ExtendedObjectsStamped


class BBox2ExtendedObjects:

    def __init__(self, _in, _out):
        rospy.init_node('clf_perception_vision_box2objects', anonymous=True)
        self.sub = rospy.Subscriber(str(_in), BoundingBoxes, self.box_cb, queue_size=1)
        self.pub = rospy.Publisher(str(_out), ExtendedObjects, queue_size=1)
        rospy.logdebug(">>> Box2People is ready.")

    def box_cb(self, data):
        try:
            e = ExtendedObjects()
            h = std_msgs.msg.Header()
            h.stamp = rospy.Time.now()
            e.header = h
            for item in data.boundingBoxes:
                if item.Class == "person":
                    continue
                else:
                    if item.probability > 0.3:
                        o = ExtendedObjectsStamped()
                        o.header = h
                        o.bbox_xmin = item.xmin
                        o.bbox_xmax = item.xmax
                        o.bbox_ymin = item.ymin
                        o.bbox_ymax = item.ymax
                        o.probability = float('%.2f' % item.probability)
                        o.category = item.Class
                        e.objects.append(o)
            self.pub.publish(e)
        except Exception, ex:
            rospy.logwarn(">>> Problem receiving data, %s" % str(ex))


if __name__ == "__main__":

    parser = OptionParser()
    parser.add_option("--intopic", dest="intopic", default="/darknet_ros/bounding_boxes")
    parser.add_option("--outtopic", dest="outtopic", default="/clf_perception_vision/objects/raw")
    (options, args) = parser.parse_args()
    B2P = BBox2ExtendedObjects(options.intopic, options.outtopic)
    rate = rospy.Rate(2.0)
    while not rospy.is_shutdown():
        try:
            rospy.spin()
        except rospy.ROSInterruptException, ex:
            rospy.logwarn(">>> Exiting, %s" % str(ex))

