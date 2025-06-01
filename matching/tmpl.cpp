#include "tmpl.h"

#include "opencv2/imgproc.hpp"

Tmpl::Tmpl()
    : m_isPatternLearned(false),
    m_borderColor(0),
    m_minReduceArea(0),
    m_topLayer(0) {

}

void Tmpl::learn(cv::Mat &img, int min_length) {
  m_pattern_image = img.clone();
  clearPattern();
  findTopLayer(min_length);

  cv::buildPyramid(img, m_vecPyramid, m_topLayer);
  m_borderColor = mean (img).val[0] < 128 ? 255 : 0;
  int pyr_size = (int)m_vecPyramid.size();
  this->resize(pyr_size);
  for (int idx=0; idx<pyr_size; idx++) {
    double invArea = 1. / ((double)m_vecPyramid[idx].rows * m_vecPyramid[idx].cols);
    Scalar templMean, templSdv; double templNorm = 0;
    // double templSum2 = 0;

    meanStdDev (m_vecPyramid[idx], templMean, templSdv);
    templNorm = templSdv[0] * templSdv[0] +
                templSdv[1] * templSdv[1] +
                templSdv[2] * templSdv[2] +
                templSdv[3] * templSdv[3];
    if (templNorm < DBL_EPSILON) {
      m_vecResultEqual1[idx] = true;
    }

    // templSum2 = templNorm + templMean[0] * templMean[0] +
    //             templMean[1] * templMean[1] +
    //             templMean[2] * templMean[2] +
    //             templMean[3] * templMean[3];
    // templSum2 /= invArea;
    templNorm = std::sqrt (templNorm);
    templNorm /= std::sqrt (invArea); // care of accuracy here
    m_vecInvArea[idx] = invArea;
    m_vecTemplMean[idx] = templMean;
    m_vecTemplNorm[idx] = templNorm;
  }
  m_isPatternLearned = true;
}

void Tmpl::clearPattern() {
  vector<cv::Mat>().swap(m_vecPyramid);
  vector<double>().swap(m_vecTemplNorm);
  vector<double>().swap(m_vecInvArea);
  vector<Scalar>().swap(m_vecTemplMean);
  vector<bool>().swap(m_vecResultEqual1);
}

void Tmpl::resize(int new_size) {
  m_vecTemplMean.resize (new_size);
  m_vecTemplNorm.resize (new_size, 0);
  m_vecInvArea.resize (new_size, 1);
  m_vecResultEqual1.resize (new_size, false);
}

void Tmpl::findTopLayer(int min_length) {
  m_topLayer = 0; m_minReduceArea = min_length * min_length;
  int area = m_pattern_image.cols * m_pattern_image.rows;
  while (area > m_minReduceArea) {
    area /= 4; m_topLayer++;
  }
}
