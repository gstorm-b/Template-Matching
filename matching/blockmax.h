#ifndef BLOCKMAX_H
#define BLOCKMAX_H

#include <vector>
#include <opencv2/core.hpp>

using namespace std;
using namespace cv;

class MatchBlock {
  public:
    MatchBlock();
    MatchBlock(Rect rect, double val_max, Point ptMaxLoc);
    ~MatchBlock();

  public:
    Rect _rect;
    double _maxValue;
    Point _pointMaxLoc;
};

class BlockMax {
  public:
    BlockMax();
    BlockMax(cv::Mat matSrc, cv::Size sizePattern);
    ~BlockMax(); void UpdateMax(Rect rectIgnore);
    void GetMaxValueLoc(double &dMax, Point& ptMaxLoc);
  private:
    vector<MatchBlock> _vecBlocks;
    cv::Mat _matSrc;
};

#endif // BLOCKMAX_H
