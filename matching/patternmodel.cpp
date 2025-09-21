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

inline std::vector<cv::Point> tNeighborSimplify(const std::vector<cv::Point>& contour, int T = 5, double tolerance = 2.0) {
  const size_t N = contour.size();
  if (N < 2 || T < 2) {
    return contour;
  }

  // Nothing to do
  std::vector<cv::Point> simplified;
  simplified.reserve(N);
  simplified.push_back(contour.front());
  // Always keep the first point
  size_t i = 0;
  while (i < N - 1) {
    bool compressed = false;
    // j goes from far neighbor (i+T) back to the immediate next (i+1)
    const size_t j_max = std::min(i + static_cast<size_t>(T), N - 1);
    for (size_t j = j_max;j > i + 0;--j) {
      const cv::Point& p1 = simplified.back();
      // start point
      const cv::Point& p2 = contour[j];
      // candidate end point
      const cv::Point2f seg = p2 - p1;
      const double seg_len = cv::norm(seg);
      if (seg_len == 0.0) {
        continue;
      } // degenerate; skip
      // Compute max distance of intermediate points k ∈ (i, j)
      double dmax = 0.0;
      for (size_t k = i + 1; k < j; ++k) {
        const cv::Point& pk = contour[k];
        // perpendicular distance from pk to line p1‑p2
        double cross = std::abs(seg.x * (pk.y - p1.y) - seg.y * (pk.x - p1.x));
        double dist = cross / seg_len;
        if (dist > dmax) {
          dmax = dist;
        }
        if (dmax > tolerance) {
          break;
        } // early exit
      }
      if (dmax <= tolerance) {
        // All intermediate points are close enough -> compress them
        simplified.push_back(p2);
        i = j;
        // jump forward
        compressed = true;
        break;
      }
    }
    if (!compressed) {
      // Could not compress with any neighbor → keep next point
      simplified.push_back(contour[i + 1]);
      ++i;
    }
  }

  return simplified;
}

inline std::vector<cv::Point> t_neighbor_simplify(const std::vector<cv::Point>& contour, int T = 5, double tolerance = 2.0) {
  const size_t N = contour.size();
  if (N < 2 || T < 2) {
    // Nothing to do
    return contour;
  }

  std::vector<cv::Point> simplified;
  simplified.reserve(N);
  simplified.push_back(contour.front()); // Always keep the first point
  size_t i = 0;
  while (i < N - 1) {
    bool compressed = false;
    // j goes from far neighbor (i+T) back to the immediate next (i+1)
    const size_t j_max = std::min(i + static_cast<size_t>(T), N - 1);
    for (size_t j = j_max;j > i + 0;--j) {
      const cv::Point& p1 = simplified.back(); // start point
      const cv::Point& p2 = contour[j]; // candidate end point
      const cv::Point2f seg = p2 - p1;
      const double seg_len = cv::norm(seg);
      if (seg_len == 0.0) {
        continue;
      } // degenerate; skip
      // Compute max distance of intermediate points k ∈ (i, j)
      double dmax = 0.0;
      for (size_t k = i + 1; k < j; ++k) {
        const cv::Point& pk = contour[k];
        // perpendicular distance from pk to line p1‑p2
        double cross = std::abs(seg.x * (pk.y - p1.y) - seg.y * (pk.x - p1.x));
        double dist = cross / seg_len;
        if (dist > dmax) {
          dmax = dist;
        }
        if (dmax > tolerance) {
          break;
        } // early exit
      }
      if (dmax <= tolerance) {
        // All intermediate points are close enough -> compress them
        simplified.push_back(p2);
        i = j;
        // jump forward
        compressed = true;
        break;
      }
    }
    if (!compressed) {
      // Could not compress with any neighbor → keep next point
      simplified.push_back(contour[i + 1]);
      ++i;
    }
  }

  return simplified;
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
      std::vector<cv::Point> resampled = t_neighbor_simplify(pattern.contours[contourIdx], 4);

      for (int pointInx = 0;pointInx < resampled.size();pointInx++) {
      // for (int pointInx = 0; pointInx < pattern.contours[contourIdx].size(); pointInx++) {
        PatternPoint tempData;
        double mag;
        // tempData.Coordinates = pattern.contours[contourIdx][pointInx];
        tempData.Coordinates = resampled[pointInx];
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

void PatternModel::setMinScore(float min_score, float greediness) {
  if(!m_hasPatternLearned) {
    return;
  }

  if (m_patterns.size() == 0) {
    return;
  }

  for (int idx=0;idx<m_patterns.size();idx++) {
    if (idx == 0) {
      m_patterns[idx].score = min_score;
    } else {
      m_patterns[idx].score = m_patterns[idx-1].score * 0.9;
    }

    PatternLayer &pattern = m_patterns.at(idx);
    vector<PatternPoint> &points = pattern.patternPoints;
    long noOfCordinates = points.size();
    double normMinScore = m_patterns[idx].score / noOfCordinates;
    double normGreediness = ((1 - greediness * m_patterns[idx].score) /
                             (1 - greediness)) / noOfCordinates;
    for (int pts_idx=0;pts_idx<noOfCordinates;pts_idx++) {
      // Early rejection (greediness)
      int sumOfCoords = pts_idx + 1;
      double minBreakScores = std::min(
          (m_patterns[idx].score - 1) + normGreediness * sumOfCoords,
          normMinScore * sumOfCoords);
      m_patterns[idx].norm_stop_early_score.push_back(minBreakScores);
    }
  }
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
