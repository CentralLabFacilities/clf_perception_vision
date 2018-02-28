//
// Created by rfeldhans on 06.02.18.
//

#ifndef CLF_PERCEPTION_VISION_CONTINUITY_H
#define CLF_PERCEPTION_VISION_CONTINUITY_H

#include "common.h"
#include <iostream>


using cv::RotatedRect;
using cv::Size2f;

namespace cmt {

    /**
     * Class to check for and balance the continuity of the tracked frame (rect) and points.
     * Sometimes, when the CMT looses most of its features and fails to re-aquire them, its better to just search for
     * the whole ROI again. This class shall compute if this is the case.
     * It's main function is the check_for_continuity function which will determine if the continuity is broken based
     * on the direction and length of jumps of the tracking rect in the last frames as well as the found feature points.
     */
    class Continuity {

    public:
        Continuity() : max_saved_rect_movements(5), cycles_to_skip(0) {};

        void initialize(const vector<Point2f> points, const RotatedRect rect);

        bool check_for_continuity(const vector<Point2f> points, const RotatedRect rect);

    private:
        ///////variables

        //momentary and previous rect
        RotatedRect tracking_rect;
        RotatedRect tracking_rect_prev;

        //momentary and previous points which where found by the base cmt
        vector<Point2f> tracking_points;
        vector<Point2f> tracking_points_prev;

        //number of initial points
        unsigned int initial_amount_points;
        //max of width, height of the initial rect. used to normalize the movement rating in generate_movement_rating.
        float initial_rect_size;
        //last known rect position.
        Point2f last_rect_position;
        //last movements will also be saved. Note that the movements are saved in the form x=distance, y=angle,
        //where the angle is in regards to the last angle and the first angle is always 0. Will not contain more than...
        vector<Point2f> last_rect_movements;
        //...this many positions (aka the continuity will only remember this many positions)
        unsigned int max_saved_rect_movements;
        //keeps track on how often continuity calculation should be skipped, after continuity is broken
        int cycles_to_skip;

        //////functions

        void get_center_of_rect(const RotatedRect rect, Point2f& point);

        void add_new_movement_point(const Point2f &new_position);

        void generate_movement_rating(float &rating);

        void normalize_point(const Point2f &orig, Point2f &normalized);
    };
}

#endif //CLF_PERCEPTION_VISION_CONTINUITY_H
