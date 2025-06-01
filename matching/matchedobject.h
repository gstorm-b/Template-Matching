#ifndef MATCHEDOBJECT_H
#define MATCHEDOBJECT_H

#include <opencv2/core.hpp>

using namespace cv;

class MatchedObject {
  public:
    MatchedObject();

  public:
    Point2f point_LT;
    Point2f point_RT;
    Point2f point_LB;
    Point2f point_RB;
    Point2f point_Center;
    double matched_Angle;
    double matched_Score;
};

#endif // MATCHEDOBJECT_H
