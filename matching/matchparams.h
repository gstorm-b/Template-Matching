#ifndef MATCHPARAMS_H

#define MATCHPARAMS_H
#include <opencv2/core.hpp>

using namespace cv;

class MatchParams {
  public:
    MatchParams();
    MatchParams(Point2f ptMinMax, double score, double angle);
    ~MatchParams();

  public:
    Point2d _point;
    double _matchScore;
    double _matchAngle;
    // Mat _matRotatedSrc;
    Rect _rectRoi;
    double _angleStart;
    double _angleEnd;
    RotatedRect _rectR;
    Rect _rectBounding;
    bool _delete;
    // for subpixel
    double _vecResult[3][3];
    int _maxScoreIndex;
    bool _posOnBorder;
    Point2d _pointSubPixel;
    double _newAngle;
};

#endif // MATCHPARAMS_H

