#!/usr/bin/env python

# Author: Felix Friese [ffriese AT techfak.uni-bielefeld DOT de], Florian Lier [flier AT techfak.uni-bielefeld DOT de]

import time
import rospy
import random
from threading import Lock
from optparse import OptionParser
from bayes_people_tracker_msgs.msg import PersonImage, PeopleTrackerImage
from clf_perception_vision_msgs.srv import DoIKnowThatPersonImage, LearnPersonImage, LearnPersonImageRequest


class SendSurf:
    def __init__(self, _in):
        self.l = Lock()
        self.current_image = None
        rospy.init_node('clf_perception_send_surf', anonymous=True)
        self.sub = rospy.Subscriber(str(_in), PeopleTrackerImage, self.person_cb, queue_size=1)
        self.default_names = ["Charlie", "Skyler", "Frankie", "Charlie", "Dakota", "Finley", "Rory",
                              "Jamie", "Jordan", "Hayden", "Payton", "Robin", "Cameron", "Ash",
                              "Bobby", "Fred", "Gabby", "Jacky"]
        rospy.loginfo(">>> Waiting for Service ... %s" % "clf_interactive_surb_learn")
        rospy.wait_for_service('clf_interactive_surb_learn')
        self.service = rospy.ServiceProxy('clf_interactive_surb_learn', LearnPersonImage)
        rospy.loginfo(">>> SendSurf is ready.")

    def person_cb(self, data):
        self.l.acquire()
        try:
            for p in data.trackedPeopleImg:
                self.current_image = p.image
        except Exception, ex:
            self.l.release()
            rospy.logwarn(">>> %s" % str(ex))
        self.l.release()

    def call_service(self):
        self.l.acquire()
        if self.current_image is not None:
            req = LearnPersonImageRequest()
            req.roi = self.current_image
            req.name = random.choice(self.default_names)
            resp = self.service(req)
            rospy.loginfo(">>> Success: %s" % str(resp.success))
        else:
            rospy.logwarn(">>> Could not send request, image was emtpy")
        self.l.release()

if __name__ == "__main__":

    parser = OptionParser()
    parser.add_option("--intopic", dest="intopic", default="/people_tracker/people/extended")
    (options, args) = parser.parse_args()
    send_surf = SendSurf(options.intopic)
    while not rospy.is_shutdown():
        try:
            time.sleep(0.1)
            inp = raw_input(">>> Press ENTER to send learning data")
            send_surf.call_service()
        except rospy.ROSInterruptException, ex:
            rospy.logwarn(">>> Exiting, %s" % str(ex))
