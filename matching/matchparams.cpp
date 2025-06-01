#include "matchparams.h"

MatchParams::MatchParams() {
  _matchScore = 0.0;
  _matchAngle = 0.0;
}

MatchParams::MatchParams(Point2f ptMinMax, double score, double angle) {
  _point = ptMinMax;
  _matchScore = score;
  _matchAngle = angle;
  _delete = false;
  _newAngle = 0.0;
  _posOnBorder = false;
}

MatchParams::~MatchParams() {

}
