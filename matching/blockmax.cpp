#include "blockmax.h"

MatchBlock::MatchBlock() {
  _maxValue = 0.0; }

MatchBlock::MatchBlock(Rect rect, double val_max, Point ptMaxLoc) {
  _rect = rect;
  _maxValue = val_max;
  _pointMaxLoc = ptMaxLoc;
}

MatchBlock::~MatchBlock() {

}

BlockMax::BlockMax() {

}

BlockMax::BlockMax(cv::Mat matSrc, cv::Size sizePattern) {
  this->_matSrc = matSrc; int iBlockW = sizePattern.width * 2;
  int iBlockH = sizePattern.height * 2;
  int iCol = _matSrc.cols / iBlockW;
  bool bHResidue = _matSrc.cols % iBlockW != 0;
  int iRow = _matSrc.rows / iBlockH;
  bool bVResidue = _matSrc.rows % iBlockH != 0;

  if (iCol == 0 || iRow == 0) {
    _vecBlocks.clear ();
    return;
  }
  _vecBlocks.resize(iCol * iRow);
  int iCount = 0;
  for (int y = 0; y < iRow ; y++) {
    for (int x = 0; x < iCol; x++) {
      Rect rectBlock (x * iBlockW, y * iBlockH, iBlockW, iBlockH);
      _vecBlocks[iCount]._rect = rectBlock;
      minMaxLoc(_matSrc(rectBlock), 0, &_vecBlocks[iCount]._maxValue, 0, &_vecBlocks[iCount]._pointMaxLoc);
      _vecBlocks[iCount]._pointMaxLoc += rectBlock.tl ();
      iCount++;
    }
  }

  if (bHResidue && bVResidue) {
    Rect rectRight (iCol * iBlockW, 0, _matSrc.cols - iCol * iBlockW, _matSrc.rows);
    MatchBlock blockRight; blockRight._rect = rectRight;
    minMaxLoc(_matSrc(rectRight), 0, &blockRight._maxValue, 0, &blockRight._pointMaxLoc);
    blockRight._pointMaxLoc += rectRight.tl (); _vecBlocks.push_back (blockRight);
    Rect rectBottom (0, iRow * iBlockH, iCol * iBlockW, _matSrc.rows - iRow * iBlockH);
    MatchBlock blockBottom;
    blockBottom._rect = rectBottom;
    minMaxLoc(_matSrc(rectBottom), 0, &blockBottom._maxValue, 0, &blockBottom._pointMaxLoc);
    blockBottom._pointMaxLoc += rectBottom.tl ();
    _vecBlocks.push_back (blockBottom);
  } else if (bHResidue) {
    Rect rectRight (iCol * iBlockW, 0, _matSrc.cols - iCol * iBlockW, _matSrc.rows);
    MatchBlock blockRight;
    blockRight._rect = rectRight;
    minMaxLoc(_matSrc(rectRight), 0, &blockRight._maxValue, 0, &blockRight._pointMaxLoc);
    blockRight._pointMaxLoc += rectRight.tl ();
    _vecBlocks.push_back (blockRight);
  } else {
    Rect rectBottom (0, iRow * iBlockH, _matSrc.cols, _matSrc.rows - iRow * iBlockH);
    MatchBlock blockBottom;
    blockBottom._rect = rectBottom;
    minMaxLoc(_matSrc(rectBottom), 0, &blockBottom._maxValue, 0, &blockBottom._pointMaxLoc);
    blockBottom._pointMaxLoc += rectBottom.tl ();
    _vecBlocks.push_back(blockBottom);
  }
}

BlockMax::~BlockMax() {

}

void BlockMax::UpdateMax(Rect rectIgnore) {
  if (_vecBlocks.size () == 0) {
    return;
  }
  // Find all blocks that intersect with rectIgnore
  int iSize = (int)_vecBlocks.size();
  for (int i = 0; i < iSize ; i++) {
    Rect rectIntersec = rectIgnore & _vecBlocks[i]._rect;
    // No intersection
    if (rectIntersec.width == 0 && rectIntersec.height == 0) {
      continue;
    }
    // If there is an intersection, update the extreme value and extreme value position
    minMaxLoc(_matSrc(_vecBlocks[i]._rect), 0, &_vecBlocks[i]._maxValue, 0, &_vecBlocks[i]._pointMaxLoc);
    _vecBlocks[i]._pointMaxLoc += _vecBlocks[i]._rect.tl();
  }
}

void BlockMax::GetMaxValueLoc(double &dMax, Point &ptMaxLoc) {
  int iSize = (int)_vecBlocks.size ();
  if (iSize == 0) {
    cv::minMaxLoc(_matSrc, 0, &dMax, 0, &ptMaxLoc);
    return;
  }
  // Find the maximum value from the block
  int iIndex = 0; dMax = _vecBlocks[0]._maxValue;
  for (int i = 1 ; i < iSize; i++) {
    if (_vecBlocks[i]._maxValue >= dMax) {
      iIndex = i; dMax = _vecBlocks[i]._maxValue;
    }
  }
  ptMaxLoc = _vecBlocks[iIndex]._pointMaxLoc;
}
