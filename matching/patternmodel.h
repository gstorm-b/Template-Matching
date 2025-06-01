#ifndef PATTERNMODEL_H
#define PATTERNMODEL_H

#include <vector>
#include <opencv2/core.hpp>

using namespace std;
using namespace cv;

class EdgeProfile {
  public:
    EdgeProfile();
    ~EdgeProfile();

  public:
    double threshLower;
    double threshUpper;
    int kernelSize;
    cv::Size blurKernelSize;
    double greediness;
};

class PatternModel {
  public:
    struct PatternPoint {
      Point2d Coordinates;
      Point2d Derivative;
      double Angle;
      double Magnitude;
      Point2d Center;
      Point2d Offset;
    };

    struct PatternLayer {
      cv::Mat image;
      vector<PatternPoint> patternPoints;
      vector<vector<Point>> contours;
      vector<Vec4i> hierarchies;
    };

    PatternModel();
    void setEdgeProfile(EdgeProfile &profile);
    bool learnPattern(cv::Mat &imgPattern, int min_length);

    inline bool isPatternLearned() {
      return m_hasPatternLearned;
    }

    inline const vector<cv::Mat>* const getPyramid() const {
      return &m_pyramids;
    }

    inline const vector<PatternLayer>* const getPatterns() const {
      return &m_patterns;
    }

    inline Mat getPatternImage() {
      return m_imageRaw.clone();
    }

    inline int getTopLayer() {
      return m_topLayer;
    }

  private:
    void clearPatternData();
    void findTopLayer();
    // void learnSingleLayer();

  public:
    double min_score;

  private:
    cv::Mat m_imageRaw;
    bool m_hasPatternLearned;
    int m_minReduceLength;
    int m_minReduceArea;
    int m_topLayer;
    int m_borderColor;
    // vector<PatternPoint> m_patternOfModel;
    // vector<vector<Point>> m_contoursOfModel;
    // vector<Vec4i> m_hierarchyOfContours;
    vector<Mat> m_pyramids;
    vector<PatternLayer> m_patterns;
    EdgeProfile m_edgeProfile;
};

#endif // PATTERNMODEL_H
