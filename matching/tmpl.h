#ifndef TMPL_H
#define TMPL_H

#include <vector>
#include <opencv2/core.hpp>

using namespace std;
using namespace cv;

class Tmpl {
  public:
    Tmpl();
    void learn(cv::Mat &img, int min_length);

    inline bool isPatternLearned() {
      return m_isPatternLearned;
    }

    inline int patternTopLayer() {
      return m_topLayer;
    }

    inline int borderColor() {
      return m_borderColor;
    }

    inline const vector<Mat>* const getPyramid() const {
      return &m_vecPyramid;
    }

    inline const vector<Scalar>* const getTemplMean() const {
      return &m_vecTemplMean;
    }

    inline const vector<double>* const getTemplNorm() const {
      return &m_vecTemplNorm;
    }

    inline const vector<double>* const getInvArea() const {
      return &m_vecInvArea;
    }

    inline const vector<bool>* const getResultEqual1() const {
      return &m_vecResultEqual1;
    }

    inline Mat getPatternImage() {
      return m_pattern_image.clone();
    }

  private:
    void clearPattern();
    void resize(int new_size);
    void findTopLayer(int min_length);


  private:
    Mat m_pattern_image;
    vector<Mat> m_vecPyramid;
    vector<Scalar> m_vecTemplMean;
    vector<double> m_vecTemplNorm;
    vector<double> m_vecInvArea;
    vector<bool> m_vecResultEqual1;
    bool m_isPatternLearned;
    int m_borderColor;
    int m_minReduceArea;
    int m_topLayer;
};

#endif // TMPL_H
