#include "patternmodel.h"
#include "opencv2/imgproc.hpp"

EdgeProfile::EdgeProfile()
    : threshLower(100.0),
    threshUpper(200.0),
    kernelSize(3),
    blurKernelSize(5, 5),
    greediness(0.0) {

}

EdgeProfile::~EdgeProfile() {

}

PatternModel::PatternModel()
    : m_hasPatternLearned(false) {

}

void PatternModel::setEdgeProfile(EdgeProfile &profile) {
  this->m_edgeProfile = profile;
}

bool PatternModel::learnPattern(cv::Mat &imgPattern, int min_length) {
  m_hasPatternLearned = false;
  if (imgPattern.empty()) {
    return false;
  }

  m_imageRaw = imgPattern.clone();
  clearPatternData();
  m_minReduceLength = min_length;
  this->findTopLayer();
  cv::buildPyramid(m_imageRaw, m_pyramids, m_topLayer);
  m_borderColor = mean(imgPattern).val[0] < 128 ? 255 : 0;
  int pyr_size = (int)m_pyramids.size(); m_patterns.resize(pyr_size);
  for (int index=0;index<pyr_size;index++) {
    PatternLayer pattern;
    pattern.patternPoints.clear();
    pattern.contours.clear();
    pattern.hierarchies.clear();

    Mat imgGray;
    Mat imgOutput;
    Mat gx, gy;
    Mat magnitude, angle;
    Point2d sumPoint = Point2d(0, 0);

    // cv::cvtColor(m_pyramids.at(index), imgGray, cv::COLOR_RGB2GRAY);
    imgGray = m_pyramids.at(index).clone();
    GaussianBlur(imgGray, imgGray, m_edgeProfile.blurKernelSize, 0);
    Canny(imgGray, imgOutput, m_edgeProfile.threshLower, m_edgeProfile.threshUpper,
          m_edgeProfile.kernelSize);

    // find contours
    findContours(imgOutput, pattern.contours, pattern.hierarchies,
                 cv::RETR_TREE, cv::CHAIN_APPROX_NONE);
    // use the sobel filter on the template image
    // which returns the gradients in the X (Gx) and Y (Gy) direction
    Sobel(imgGray, gx, CV_64F, 1, 0, 3);
    Sobel(imgGray, gy, CV_64F, 0, 1, 3);
  //compute the magnitude and direction(radians)
    cartToPolar(gx, gy, magnitude, angle);
    for (int contourIdx = 0; contourIdx < pattern.contours.size(); contourIdx++) {
      for (int pointInx = 0; pointInx < pattern.contours[contourIdx].size(); pointInx++) {
        PatternPoint tempData;
        double mag;
        tempData.Coordinates = pattern.contours[contourIdx][pointInx];
        tempData.Derivative = Point2d(gx.at<double>(tempData.Coordinates),
                                      gy.at<double>(tempData.Coordinates));
        tempData.Angle = angle.at<double>(tempData.Coordinates);
        mag = magnitude.at<double>(tempData.Coordinates);
        tempData.Magnitude = (mag == 0) ? 0 : (1 / mag);

        // push to container
        pattern.patternPoints.push_back(tempData);
        sumPoint += tempData.Coordinates;
      }
    }

    Point2f templateCenterPoint(sumPoint.x / pattern.patternPoints.size(),
                                sumPoint.y / pattern.patternPoints.size());

    for (PatternPoint& pointTemp : pattern.patternPoints) {
      pointTemp.Center = templateCenterPoint;
      pointTemp.Offset = pointTemp.Coordinates - pointTemp.Center;
    }
    pattern.image = m_pyramids.at(index).clone();
    m_patterns[index] = pattern;
  }
  m_hasPatternLearned = true; return true;
}

void PatternModel::clearPatternData() {
  m_patterns.clear();
}

void PatternModel::findTopLayer() {
  m_topLayer = 0; m_minReduceArea = m_minReduceLength * m_minReduceLength;
  int area = m_imageRaw.cols * m_imageRaw.rows;
  while (area > m_minReduceArea) {
    area /= 4; m_topLayer++;
  }
}
