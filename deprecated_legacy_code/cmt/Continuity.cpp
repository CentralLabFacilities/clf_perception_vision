//
// Created by rfeldhans on 06.02.18.
//
#include "Continuity.h"
#include <math.h>

using std::max;

/**
 * For a general idea of what this class does, please see the header.
 */

namespace cmt {

    /**
     * Initialize the Continuity. needed to have previous data for the iterative step.
     * @param points
     * @param rect
     */
    void Continuity::initialize(const vector<Point2f> points, const RotatedRect rect){
        // initialize previous values for main loop to work
        this->tracking_rect_prev = rect;
        this->tracking_points_prev = points;
        this->initial_amount_points = points.size()+1;

        Point2f center;
        this->get_center_of_rect(rect, center);
        this->last_rect_position = center;

        this->initial_rect_size = max(rect.size.height, rect.size.width);
    }

    /**
     * Iterative step for every frame of the tracking.
     * @param points
     * @param rect
     * @return true when continuity is preserved, false when tracked points should be reaquired
     */
    bool Continuity::check_for_continuity(const vector<Point2f> points, const RotatedRect rect){
        bool continuity_preserved = true;
        this->tracking_rect = rect;
        this->tracking_points = points;

        Point2f center;
        this->get_center_of_rect(rect, center);
        this->add_new_movement_point(center);


        //TODO work with tracked points
        float movement_rating;
        this->generate_movement_rating(movement_rating);
        std::cout << "movement rating: " << movement_rating << std::endl;

        if(movement_rating > 0.2 || movement_rating < 0.00005){
            continuity_preserved = false;
            std::cout << "broken because of movement rating" << std::endl;
        }
        if(points.size() >= 1){
            float fraction = points.size()/(float)this->initial_amount_points;
            std::cout << "fraction " << fraction << "(" << points.size() << "/" << this->initial_amount_points << ")" << std::endl;
            if(fraction < 0.5) {
                continuity_preserved = false;
            }
        }


        this->tracking_points_prev = this->tracking_points;
        this->tracking_rect_prev = this->tracking_rect;

        return continuity_preserved;
    }

    /**
     * Helper function which calculates the center of a cv::RotatedRect.
     * @param rect the rect of which the center shall be calculated.
     * @param point a reference to the center of the given rect, which will be fileld by this function.
     */
    void Continuity::get_center_of_rect(const RotatedRect rect, Point2f& point){
        Point2f vertices[4];
        rect.points(vertices);
        for (int i = 0; i < 4; i++) {
            point.x += vertices[i].x;
            point.y += vertices[i].y;
        }
        point.x /= 4.0;
        point.y /= 4.0;
    }

    /**
     *
     * @param new_position
     */
    void Continuity::add_new_movement_point(const Point2f &new_position){
        Point2f diff = new_position - this->last_rect_position;
        Point2f movement;
        movement.x = sqrt(pow(diff.x,2)+pow(diff.y,2));//in pixels (more or less)
        movement.y = atan2( - diff.y, - diff.x) * 180 / M_PI;//in degree
        this->last_rect_movements.push_back(movement);
        if(this->last_rect_movements.size() > this->max_saved_rect_movements){
            this->last_rect_movements.erase(this->last_rect_movements.begin());
        }
        this->last_rect_position = new_position;
    }

    /**
     * Generates a rating of the movement of the rect the cmt tracks. This rating indicates how much the rect is jumping
     * around. The rating is normalized between 0.0 and 1.0. Smaller is better (aka less jumpy).
     *
     * @param rating the rating this  function will generate
     */
    void Continuity::generate_movement_rating(float &rating){
        // calculate difference in angles
        float angle_diffs = 0.0;
        for(int i = 1; i < this->last_rect_movements.size(); i++){
            float diff = this->last_rect_movements[i].y - this->last_rect_movements[i-1].y;
            if(diff > 180){
                diff -= 360.0;
            }
            else if (diff < -180){
                diff += 360.0;
            }
            angle_diffs += abs(diff);
        }
        //angle_diffs basically describes how much the direction has changed over time, based on a fraction.
        angle_diffs /= 180*this->last_rect_movements.size();

        float distance = 0.0;
        for(int i = 0; i < this->last_rect_movements.size(); i++){
            distance += this->last_rect_movements[i].x / this->last_rect_movements.size();
        }
        distance /= this->initial_rect_size;

        rating = angle_diffs * distance;
    }
}

