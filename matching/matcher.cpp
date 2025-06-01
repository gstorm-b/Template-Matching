#include "matcher.h"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <chrono>
#include <algorithm>
#include <vector>
#include <cmath>
#include <immintrin.h>
#include <omp.h>

inline int _mm_hsum_epi32(__m128i V) {
  // V3 V2 V1 V0
  // The actual speed is faster, and _mm_extract_epi32 is the slowest.
  // V3+V1 V2+V0 V1 V0
  __m128i T = _mm_add_epi32(V, _mm_srli_si128(V, 8));
  // V3+V1+V2+V0 V2+V0+V1 V1+V0 V0
  T = _mm_add_epi32(T, _mm_srli_si128(T, 4));
  // Extract low bit
  return _mm_cvtsi128_si32(T);
}

inline int IM_Conv_SIMD (unsigned char* pCharKernel, unsigned char *pCharConv, int iLength) {
  const int iBlockSize = 16, Block = iLength / iBlockSize;
  __m128i SumV = _mm_setzero_si128 ();
  __m128i Zero = _mm_setzero_si128 ();
  for (int Y = 0; Y < Block * iBlockSize; Y += iBlockSize) {
    __m128i SrcK = _mm_loadu_si128 ((__m128i*)(pCharKernel + Y));
    __m128i SrcC = _mm_loadu_si128 ((__m128i*)(pCharConv + Y));
    __m128i SrcK_L = _mm_unpacklo_epi8 (SrcK, Zero);
    __m128i SrcK_H = _mm_unpackhi_epi8 (SrcK, Zero);
    __m128i SrcC_L = _mm_unpacklo_epi8 (SrcC, Zero);
    __m128i SrcC_H = _mm_unpackhi_epi8 (SrcC, Zero);
    __m128i SumT = _mm_add_epi32 (_mm_madd_epi16 (SrcK_L, SrcC_L), _mm_madd_epi16 (SrcK_H, SrcC_H));
    SumV = _mm_add_epi32 (SumV, SumT);
  }

  int Sum = _mm_hsum_epi32 (SumV);
  for (int Y = Block * iBlockSize; Y < iLength; Y++) {
    Sum += pCharKernel[Y] * pCharConv[Y];
  }
  return Sum;
}

bool compareScoreBig2Small(const MatchParams &lhs, const MatchParams &rhs) {
  return lhs._matchScore > rhs._matchScore;
}

bool comparePtWithAngle(const pair<Point2f, double> lhs, const pair<Point2f, double> rhs) {
  return lhs.second < rhs.second;
}

Matcher::Matcher() {

}

void Matcher::setEdgePatternImage(cv::Mat &img_pattern) {
  m_edgePattern.setEdgeProfile(m_edgeProfile); m_edgePattern.learnPattern(img_pattern, min_reduce_length);
}

void Matcher::setPatternImage(cv::Mat &img_pattern) {
  m_pattern.learn(img_pattern, min_reduce_length);
}

void Matcher::setMatchSourceImage(cv::Mat &img) {
  m_img_source = img.clone();
}

bool Matcher::Match() {
  if (!m_pattern.isPatternLearned()) {
    return false;
  }

  /// Starts condition
  cv::Mat img_dst = m_pattern.getPatternImage();

  if (m_img_source.empty () || img_dst.empty ()) {
    return false;
  }
  if ((img_dst.cols < m_img_source.cols && img_dst.rows > m_img_source.rows) ||
      (img_dst.cols > m_img_source.cols && img_dst.rows < m_img_source.rows)) {
    return false;
  } if (img_dst.size().area () > m_img_source.size().area ()) {
    return false;
  }
  /// END Starts condition

  // reserve: add chrono clock marker
  std::chrono::time_point start_point = std::chrono::high_resolution_clock::now();

  /// Build source image pyramid
  // reserve: add bitwise mat
  int top_layer = m_pattern.patternTopLayer();
  std::cout << "Top layer: " << top_layer << std::endl;
  vector<Mat> vecSrcPyr;
  cv::buildPyramid(m_img_source, vecSrcPyr, top_layer);
  /// END Build source image pyramid

  /// Matching data prepare
  const vector<cv::Mat>* tmpl_vecPyramid = m_pattern.getPyramid();
  double angleStep = atan(2.0 / max(tmpl_vecPyramid->at(top_layer).cols, tmpl_vecPyramid->at(top_layer).rows)) * R2D;
  vector<double> vecAngles;
  if (tolerance_angle < VISION_TOLERANCE) {
    vecAngles.push_back (0.0);
  } else {
    for (double angle = 0; angle< (tolerance_angle + angleStep); angle += angleStep) {
      vecAngles.push_back(angle);
    }
    for (double angle = -angleStep; angle > (-tolerance_angle - angleStep); angle -= angleStep) {
    vecAngles.push_back(angle);
    }
  }

  int top_srcWidth = vecSrcPyr[top_layer].cols;
  int top_srcHeight = vecSrcPyr[top_layer].rows;
  Point2f top_layerCenter((top_srcWidth - 1) / 2.0f, (top_srcHeight - 1) / 2.0f);
  // reduce min score for each lower pyramid image
  int numOf_angles = (int)vecAngles.size();
  vector<double> vecLayerScores(top_layer+ 1, min_score);
  for (int layerIdx=1;layerIdx<=top_layer;layerIdx++) {
    vecLayerScores[layerIdx] = vecLayerScores[layerIdx - 1] * 0.9;
  }
  cv::Size top_patternMatSize = tmpl_vecPyramid->at(top_layer).size();
  bool calMaxByBlock = (vecSrcPyr[top_layer].size().area() / top_patternMatSize.area() > 500) && (max_pos_num > 10);
  vector<MatchParams> vecMatchParams; vecMatchParams.clear();
  /// END Matching data prepare

  /// Smallest layer matching
  for (int idx=0;idx<numOf_angles;idx++) {
    Mat matRotatedSrc, matR = cv::getRotationMatrix2D(top_layerCenter, vecAngles[idx], 1.0);
    Mat matResult;
    Point ptMaxLoc;
    double value;
    double maxValue;
    Size sizeBest = GetBestRotationSize(vecSrcPyr[top_layer].size(), tmpl_vecPyramid->at(top_layer).size(), vecAngles[idx]);
    float fTranslationX = (sizeBest.width - 1) / 2.0f - top_layerCenter.x;
    float fTranslationY = (sizeBest.height - 1) / 2.0f - top_layerCenter.y;
    matR.at<double> (0, 2) += fTranslationX;
    matR.at<double> (1, 2) += fTranslationY;
    warpAffine(vecSrcPyr[top_layer], matRotatedSrc, matR, sizeBest, INTER_LINEAR, BORDER_CONSTANT, Scalar(m_pattern.borderColor()));
    cv::Mat tmpl_top_mat = tmpl_vecPyramid->at(top_layer).clone();
    // cv::matchTemplate(matRotatedSrc, tmpl_top_mat, matResult, cv::TM_CCOEFF_NORMED);
    this->MatchPattern(matRotatedSrc, &m_pattern, matResult, top_layer, false);
    // this->MatchEdgePattern(matRotatedSrc, &m_edgePattern, matResult, top_layer, vecLayerScores[top_layer]);
    if (calMaxByBlock) {
      BlockMax blockMax(matResult, tmpl_vecPyramid->at(top_layer).size()); blockMax.GetMaxValueLoc(maxValue, ptMaxLoc);
      if (maxValue < vecLayerScores[top_layer]) {
        continue;
      }

      vecMatchParams.push_back(MatchParams(Point2f(ptMaxLoc.x - fTranslationX, ptMaxLoc.y - fTranslationY), maxValue, vecAngles[idx]));
      for (int j = 0; j < max_pos_num + MATCH_CANDIDATE_NUM - 1; j++) {
        ptMaxLoc = GetNextMaxLoc(matResult, ptMaxLoc, tmpl_vecPyramid->at(top_layer).size(), value, max_overlap, blockMax);

        if (value < vecLayerScores[top_layer])
          break;

        vecMatchParams.push_back(MatchParams(Point2f (ptMaxLoc.x - fTranslationX, ptMaxLoc.y - fTranslationY), value, vecAngles[idx]));
      }
    } else {
      cv::minMaxLoc(matResult, 0, &maxValue, 0, &ptMaxLoc);
      if (maxValue < vecLayerScores[top_layer]) {
        continue;
      }
      vecMatchParams.push_back(MatchParams(Point2f(ptMaxLoc.x - fTranslationX, ptMaxLoc.y - fTranslationY), maxValue, vecAngles[idx]));
      for (int j = 0; j < max_pos_num + MATCH_CANDIDATE_NUM - 1; j++) {
        ptMaxLoc = GetNextMaxLoc(matResult, ptMaxLoc, tmpl_vecPyramid->at(top_layer).size(), value, max_overlap);
        if (value < vecLayerScores[top_layer]) {
          break;
        }
        vecMatchParams.push_back(MatchParams(Point2f(ptMaxLoc.x - fTranslationX, ptMaxLoc.y - fTranslationY), value, vecAngles[idx]));
      }
    }
  // double max_loc;
  // cv::minMaxLoc(matResult, nullptr, &max_loc, nullptr, nullptr);
  // // if (max_loc < vecLayerScores[top_layer]) {
  // // continue;
  // // }
  // std::cout << "Max loc: " << max_loc << std::endl;
  // std::string relative_path = "C:/DGB/Pick_and_place/Pc_based_software/Qt/Opencv_Console/result images/";
  // std::string save_name = relative_path + "idx_" + std::to_string(idx) + "_" + std::to_string(vecAngles[idx]) + ".bmp";
  // cv::imwrite(save_name, matRotatedSrc);
  // std::string save_name1 = relative_path + "idx_" + std::to_string(idx) + "_" + std::to_string(vecAngles[idx]) + "result.bmp";
  // cv::Mat result_write;
  // // cv::cvtColor(matResult, result_write, cv::color_)
  // cv::imwrite(save_name1, matResult * 255.0);
  }
  sort(vecMatchParams.begin (), vecMatchParams.end (), compareScoreBig2Small);
  // console debug show
  // for (int idx=0;idx<vecMatchParams.size();idx++) {
  // std::cout << "Match params at: " << idx
  //           << ", Score: " << vecMatchParams[idx]._matchScore
  //           << ", Angle: " << vecMatchParams[idx]._matchAngle
  //           << ", Point: " << vecMatchParams[idx]._point << std::endl;
  // }
  /// END Smallest layer matching

  int stop_layer = (stop_layer_1) ? 1 : 0;
  vector<MatchParams> vecAllResult;
  for (int i = 0; i < (int)vecMatchParams.size (); i++) {
    double rAngle = -vecMatchParams[i]._matchAngle * D2R;
    Point2f ptLT = ptRotatePt2f(vecMatchParams[i]._point, top_layerCenter, rAngle);
    double _angleStep = atan(2.0/max(top_srcWidth, top_srcHeight)) * R2D;
    vecMatchParams[i]._angleStart = vecMatchParams[i]._matchAngle - _angleStep;
    vecMatchParams[i]._angleEnd = vecMatchParams[i]._matchAngle + _angleStep;
    if (top_layer <= stop_layer) {
      vecMatchParams[i]._point = Point2d(ptLT * ((top_layer == 0) ? 1 : 2));
      vecAllResult.push_back(vecMatchParams[i]);
    } else {
      for (int iLayer = top_layer - 1; iLayer >= stop_layer; iLayer--) {
        // Search Angle
        _angleStep = atan (2.0 / max(tmpl_vecPyramid->at(iLayer).cols, tmpl_vecPyramid->at(iLayer).rows)) * R2D;
        vector<double> vecAngles;
        //double dAngleS = vecMatchParams[i].dAngleStart, dAngleE = vecMatchParams[i].dAngleEnd;
        double dMatchedAngle = vecMatchParams[i]._matchAngle;
        if (tolerance_angle < VISION_TOLERANCE) {
          vecAngles.push_back(0.0);
        } else {
          for (int i = -1; i <= 1; i++) {
            vecAngles.push_back(dMatchedAngle + _angleStep * i);
          }
        }
        Point2f ptSrcCenter((vecSrcPyr[iLayer].cols - 1) / 2.0f, (vecSrcPyr[iLayer].rows - 1) / 2.0f);
        int iSize = (int)vecAngles.size ();
        vector<MatchParams> vecNewMatchParameter(iSize);
        int iMaxScoreIndex = 0;
        double dBigValue = -1;
        for (int j = 0; j < iSize; j++) {
          Mat matResult, matRotatedSrc;
          double dMaxValue = 0;
          Point ptMaxLoc; GetRotatedROI(vecSrcPyr[iLayer], tmpl_vecPyramid->at(iLayer).size(), ptLT * 2, vecAngles[j], matRotatedSrc);
          // MatchTemplate (matRotatedSrc, pTemplData, matResult, iLayer, true);
          this->MatchPattern(matRotatedSrc, &m_pattern, matResult, iLayer, true);
          // this->MatchEdgePattern(matRotatedSrc, &m_edgePattern, matResult, iLayer, vecLayerScores[iLayer]);
          //matchTemplate (matRotatedSrc, pTemplData->vecPyramid[iLayer], matResult, CV_TM_CCOEFF_NORMED);
          minMaxLoc (matResult, 0, &dMaxValue, 0, &ptMaxLoc);
          vecNewMatchParameter[j] = MatchParams(ptMaxLoc, dMaxValue, vecAngles[j]);
          if (vecNewMatchParameter[j]._matchScore > dBigValue) {
            iMaxScoreIndex = j; dBigValue = vecNewMatchParameter[j]._matchScore;
          }

          /// Sub-pixel estimation
          if (ptMaxLoc.x == 0 || ptMaxLoc.y == 0 || ptMaxLoc.x == matResult.cols - 1 || ptMaxLoc.y == matResult.rows - 1) {
            vecNewMatchParameter[j]._posOnBorder = true;
          }
          if (!vecNewMatchParameter[j]._posOnBorder) {
            for (int y = -1; y <= 1; y++)
              for (int x = -1; x <= 1; x++)
                vecNewMatchParameter[j]._vecResult[x + 1][y + 1] = matResult.at<float> (ptMaxLoc + Point (x, y));
          }
          /// END Sub-pixel estimation
        }

        if (vecNewMatchParameter[iMaxScoreIndex]._matchScore < vecLayerScores[iLayer])
          break;
        // Sub-pixel estimation
        if (sub_pixel_estimation && iLayer == 0 && (!vecNewMatchParameter[iMaxScoreIndex]._posOnBorder) && iMaxScoreIndex != 0 && iMaxScoreIndex != 2) {
          double dNewX = 0, dNewY = 0, dNewAngle = 0;
          SubPixEsimation(&vecNewMatchParameter, &dNewX, &dNewY, &dNewAngle, _angleStep, iMaxScoreIndex);
          vecNewMatchParameter[iMaxScoreIndex]._point = Point2d (dNewX, dNewY);
          vecNewMatchParameter[iMaxScoreIndex]._matchAngle = dNewAngle;
        }
        // Sub-pixel estimation
        double dNewMatchAngle = vecNewMatchParameter[iMaxScoreIndex]._matchAngle;
        // Return the coordinate system to (0, 0) when it was rotated (GetRotatedROI)
        Point2f ptPaddingLT = ptRotatePt2f (ptLT * 2, ptSrcCenter, dNewMatchAngle * D2R) - Point2f (3, 3);
        Point2f pt (vecNewMatchParameter[iMaxScoreIndex]._point.x + ptPaddingLT.x, vecNewMatchParameter[iMaxScoreIndex]._point.y + ptPaddingLT.y);
        // Re-rotate
        pt = ptRotatePt2f (pt, ptSrcCenter, -dNewMatchAngle * D2R);
        if (iLayer == stop_layer) {
          vecNewMatchParameter[iMaxScoreIndex]._point = pt * (stop_layer == 0 ? 1 : 2); vecAllResult.push_back(vecNewMatchParameter[iMaxScoreIndex]);
        } else {
          // Update MatchAngle ptLT
          vecMatchParams[i]._matchAngle = dNewMatchAngle;
          vecMatchParams[i]._angleStart = vecMatchParams[i]._matchAngle - _angleStep / 2;
          vecMatchParams[i]._angleEnd = vecMatchParams[i]._matchAngle + _angleStep / 2;
          ptLT = pt;
        }
      }
    }
  }

  FilterWithScore(&vecAllResult, min_score);
  // console debug show
  std::cout << "[Before overlap filter]" << vecAllResult.size() << std::endl;
  // for (int idx=0;idx<vecAllResult.size();idx++) {
  //  std::cout << "Match params at: " << idx
  //            << ", Score: " << vecAllResult[idx]._matchScore
  //            << ", Angle: " << vecAllResult[idx]._matchAngle
  //            << ", Point: " << vecAllResult[idx]._point << std::endl;
  // }
  // Finally, filter out the overlap
  // int iDstW = pTemplData->vecPyramid[iTopLayer].cols;
  // int iDstH = pTemplData->vecPyramid[iTopLayer].rows;
  int iDstW = tmpl_vecPyramid->at(stop_layer).cols * (stop_layer == 0 ? 1 : 2);
  int iDstH = tmpl_vecPyramid->at(stop_layer).rows * (stop_layer == 0 ? 1 : 2);
  for (int i = 0; i < (int)vecAllResult.size (); i++) {
    Point2f ptLT, ptRT, ptRB, ptLB; double dRAngle = -vecAllResult[i]._matchAngle * D2R;
    ptLT = vecAllResult[i]._point;
    ptRT = Point2f (ptLT.x + iDstW * (float)cos (dRAngle), ptLT.y - iDstW * (float)sin (dRAngle));
    ptLB = Point2f (ptLT.x + iDstH * (float)sin (dRAngle), ptLT.y + iDstH * (float)cos (dRAngle));
    ptRB = Point2f (ptRT.x + iDstH * (float)sin (dRAngle), ptRT.y + iDstH * (float)cos (dRAngle));
    // Record Rotation Rectangle
    vecAllResult[i]._rectR = RotatedRect(ptLT, ptRT, ptRB);
  }

  FilterWithRotatedRect(&vecAllResult, cv::TM_CCOEFF_NORMED, max_overlap);
  // Finally, filter out the overlap
  sort(vecAllResult.begin(), vecAllResult.end(), compareScoreBig2Small);
  std::cout << "[After overlap filter]" << vecAllResult.size() << std::endl;
  // for (int idx=0;idx<vecAllResult.size();idx++) {
  //  std::cout << "Match params at: " << idx
  //            << ", Score: " << vecAllResult[idx]._matchScore
  //            << ", Angle: " << vecAllResult[idx]._matchAngle
  //            << ", Point: " << vecAllResult[idx]._point << std::endl;
  // }
  matched_result.clear();
  int numOfMatched = vecAllResult.size();
  if (numOfMatched == 0) {
    return false;
  }

  int iW = tmpl_vecPyramid->at(0).cols, iH = tmpl_vecPyramid->at(0).rows;
  for (int idx=0;idx<numOfMatched;idx++) {
    MatchedObject matchObj;
    double dRAngle = -vecAllResult[idx]._matchAngle * D2R;
    matchObj.point_LT = vecAllResult[idx]._point;
    matchObj.point_RT = Point2d(matchObj.point_LT.x + iW * cos (dRAngle), matchObj.point_LT.y - iW * sin (dRAngle));
    matchObj.point_LB = Point2d(matchObj.point_LT.x + iH * sin (dRAngle), matchObj.point_LT.y + iH * cos (dRAngle));
    matchObj.point_RB = Point2d(matchObj.point_RT.x + iH * sin (dRAngle), matchObj.point_RT.y + iH * cos (dRAngle));
    matchObj.point_Center = Point2d((matchObj.point_LT.x + matchObj.point_RT.x + matchObj.point_RB.x + matchObj.point_LB.x) / 4,
                                    (matchObj.point_LT.y + matchObj.point_RT.y + matchObj.point_RB.y + matchObj.point_LB.y) / 4);
    matchObj.matched_Angle = -vecAllResult[idx]._matchAngle;
    matchObj.matched_Score = vecAllResult[idx]._matchScore;
    if (matchObj.matched_Angle < -180)
      matchObj.matched_Angle += 360;
    if (matchObj.matched_Angle > 180)
      matchObj.matched_Angle -= 360;

    matched_result.push_back(matchObj);
  }

  std::chrono::time_point stop_point = std::chrono::high_resolution_clock::now();
  double time_count = std::chrono::duration<double, std::milli>(stop_point - start_point).count();
  std::cout << "Matching time: " << time_count << std::endl;
  cv::Mat result_image;

  // if (m_img_source.type() != CV_8UC1) {
  //   cv::cvtColor(m_img_source, result_image, cv::COLOR_GRAY2BGR);
  // } else {
  //   result_image = m_img_source;
  // }

  cv::cvtColor(m_img_source, result_image, cv::COLOR_GRAY2BGR);

  DrawMatchResult(result_image, matched_result);
  m_matResult = result_image.clone();
  // std::string relative_path = "C:/DGB/Pick_and_place/Pc_based_software/Qt/Opencv_Console/result images/";
  // std::string save_name = relative_path + "result_image_66.bmp";
  // cv::imwrite(save_name, result_image);
  // std::cout << "Result image has written 3!" << std::endl;
  return true;
}

bool Matcher::MatchEdge() {
  // for (int idx=0;idx<m_edgePattern.getPatterns()->size();idx++) {
  // // cv::Mat imgGray, imgOutput;
  // // GaussianBlur(m_edgePattern.getPyramid()->at(idx), imgGray, cv::Size(5,5), 0);
  // // Canny(imgGray, imgOutput, 100, 200, 3);
  // std::string show_name = "Layer_dst_" + std::to_string(idx);
  // cv::imshow(show_name, m_edgePattern.getPatterns()->at(idx).image);
  // }
  // cv::Mat matResult;
  // std::chrono::time_point start_point1 = std::chrono::high_resolution_clock::now();
  // this->MatchEdgePattern(m_img_source, &m_edgePattern, matResult, 0, 0.9, true);
  // std::chrono::time_point stop_point1 = std::chrono::high_resolution_clock::now();
  // double time_count1 = std::chrono::duration<double, std::milli>(stop_point1 - start_point1).count();
  // // cv::imshow("Image1", m_img_source);
  // // cv::imshow("Image2", m_edgePattern.getPatterns()->at(2).image);
  // // cv::imshow("Image", matResult);
  // double minVal, maxVal;
  // Point ptMaxLoc;
  // cv::minMaxLoc(matResult, &minVal, &maxVal, nullptr, &ptMaxLoc);
  // std::cout << "Matching time: " << time_count1 << std::endl;
  // std::cout << minVal << " " << maxVal << " " << ptMaxLoc << std::endl;
  // // std::cout << matResult << std::endl;
  // cv::Mat matShow = m_img_source.clone();
  // cv::cvtColor(m_img_source, matShow, cv::COLOR_GRAY2BGR);
  // // Point2d center_ = m_edgePattern.getPatterns()->at(0).patternPoints.at(0).Center;
  // // Rect showrect(ptMaxLoc.x - center_.x, ptMaxLoc.y - center_.y,
  // //               m_edgePattern.getPatterns()->at(0).image.cols,
  // //               m_edgePattern.getPatterns()->at(0).image.rows);
  // Rect showrect(ptMaxLoc.x, ptMaxLoc.y, m_edgePattern.getPatterns()->at(0).image.cols,
  //               m_edgePattern.getPatterns()->at(0).image.rows);
  // cv::Scalar box_color(255, 0, 0);
  // cv::rectangle(matShow, showrect, box_color, 2, cv::LINE_AA);
  // cv::imshow("matShow", matShow);
  // cv::waitKey(0);
  // cv::destroyAllWindows();
  // return true;

  if (!m_edgePattern.isPatternLearned()) {
    return false;
  }

  /// Starts condition
  cv::Mat img_dst = m_edgePattern.getPatternImage();
  if (m_img_source.empty () || img_dst.empty ()) {
    return false;
  }
  if ((img_dst.cols < m_img_source.cols && img_dst.rows > m_img_source.rows)
      || (img_dst.cols > m_img_source.cols && img_dst.rows < m_img_source.rows)) {
    return false;
  }
  if (img_dst.size().area () > m_img_source.size().area ()) {
    return false;
  }
  /// END Starts condition

  // reserve: add chrono clock marker
  std::chrono::time_point start_point = std::chrono::high_resolution_clock::now();
  /// Build source image pyramid

  // reserve: add bitwise mat
  int top_layer = m_edgePattern.getTopLayer();

  std::cout << "Top layer: " << top_layer << std::endl;
  vector<Mat> vecSrcPyr;
  cv::buildPyramid(m_img_source, vecSrcPyr, top_layer);
  // for (int idx=0;idx<vecSrcPyr.size();idx++) {
  // cv::Mat imgGray, imgOutput;
  // GaussianBlur(vecSrcPyr[idx], imgGray, cv::Size(5,5), 0);
  // Canny(imgGray, imgOutput, 100, 200, 3);
  // std::string show_name = "Layer_" + std::to_string(idx);
  // cv::imshow(show_name, imgOutput);
  // }
  // for (int idx=0;idx<m_edgePattern.getPyramid()->size();idx++) {
  // cv::Mat imgGray, imgOutput;
  // GaussianBlur(m_edgePattern.getPyramid()->at(idx), imgGray, cv::Size(5,5), 0);
  // Canny(imgGray, imgOutput, 100, 200, 3);
  // std::string show_name = "Layer_dst_" + std::to_string(idx);
  // cv::imshow(show_name, imgOutput);
  // } // cv::waitKey(0);
  // cv::destroyAllWindows();
  // return true;

  /// Build source image pyramid
  /// Matching data prepare
  const vector<cv::Mat>* tmpl_vecPyramid = m_edgePattern.getPyramid();
  double angleStep = atan(2.0 / max(tmpl_vecPyramid->at(top_layer).cols, tmpl_vecPyramid->at(top_layer).rows)) * R2D;
  vector<double> vecAngles;
  if (tolerance_angle < VISION_TOLERANCE) {
    vecAngles.push_back (0.0);
  } else {
    for (double angle = 0; angle< (tolerance_angle + angleStep); angle += angleStep) {
      vecAngles.push_back(angle);
    }
    for (double angle = -angleStep; angle > (-tolerance_angle - angleStep); angle -= angleStep) {
      vecAngles.push_back(angle);
    }
  }

  int top_srcWidth = vecSrcPyr[top_layer].cols;
  int top_srcHeight = vecSrcPyr[top_layer].rows;
  Point2f top_layerCenter((top_srcWidth - 1) / 2.0f, (top_srcHeight - 1) / 2.0f);
  // reduce min score for each lower pyramid image
  int numOf_angles = (int)vecAngles.size();
  vector<double> vecLayerScores(top_layer+ 1, min_score);
  for (int layerIdx=1;layerIdx<=top_layer;layerIdx++) {
    vecLayerScores[layerIdx] = vecLayerScores[layerIdx - 1] * 0.98;
  }
  cv::Size top_patternMatSize = tmpl_vecPyramid->at(top_layer).size();
  bool calMaxByBlock = (vecSrcPyr[top_layer].size().area() / top_patternMatSize.area() > 500) && (max_pos_num > 10);
  vector<MatchParams> vecMatchParams;
  vecMatchParams.clear();
  /// END Matching data prepare
  /// Smallest layer matching
  for (int idx=0;idx<numOf_angles;idx++) {
    Mat matRotatedSrc, matR = cv::getRotationMatrix2D(top_layerCenter, vecAngles[idx], 1.0);
    Mat matResult;
    Point ptMaxLoc;
    double value;
    double maxValue;
    Size sizeBest = GetBestRotationSize(vecSrcPyr[top_layer].size(), tmpl_vecPyramid->at(top_layer).size(), vecAngles[idx]);
    float fTranslationX = (sizeBest.width - 1) / 2.0f - top_layerCenter.x;
    float fTranslationY = (sizeBest.height - 1) / 2.0f - top_layerCenter.y;
    matR.at<double> (0, 2) += fTranslationX;
    matR.at<double> (1, 2) += fTranslationY;
    warpAffine(vecSrcPyr[top_layer], matRotatedSrc, matR, sizeBest, INTER_LINEAR, BORDER_CONSTANT, Scalar(m_pattern.borderColor()));
    cv::Mat tmpl_top_mat = tmpl_vecPyramid->at(top_layer).clone();

#ifndef USE_EDGE_SIMD
    this->MatchEdgePattern(matRotatedSrc, &m_edgePattern, matResult, top_layer, vecLayerScores[top_layer], false);
#else
    this->MatchEdgePatternSIMD(matRotatedSrc, &m_edgePattern, matResult, top_layer, vecLayerScores[top_layer], false);
#endif

    // cv::imshow("Image", matResult);
    // cv::waitKey(0);
    // cv::imshow("Image", matResult*255);
    // cv::waitKey(0);
    // cv::destroyAllWindows();
    // return true;
    if (calMaxByBlock) {
      BlockMax blockMax(matResult, tmpl_vecPyramid->at(top_layer).size());
      blockMax.GetMaxValueLoc(maxValue, ptMaxLoc);
      if (maxValue < vecLayerScores[top_layer]) {
        continue;
      }
      vecMatchParams.push_back(MatchParams(Point2f(ptMaxLoc.x - fTranslationX, ptMaxLoc.y - fTranslationY), maxValue, vecAngles[idx]));
      for (int j = 0; j < max_pos_num + MATCH_CANDIDATE_NUM - 1; j++) {
        ptMaxLoc = GetNextMaxLoc(matResult, ptMaxLoc, tmpl_vecPyramid->at(top_layer).size(), value, max_overlap, blockMax);
        if (value < vecLayerScores[top_layer]) {
          break;
        }
        vecMatchParams.push_back(MatchParams(Point2f (ptMaxLoc.x - fTranslationX, ptMaxLoc.y - fTranslationY), value, vecAngles[idx]));
      }
    } else {
      cv::minMaxLoc(matResult, 0, &maxValue, 0, &ptMaxLoc);
      if (maxValue < vecLayerScores[top_layer]) {
        continue;
      }
      vecMatchParams.push_back(MatchParams(Point2f(ptMaxLoc.x - fTranslationX, ptMaxLoc.y - fTranslationY), maxValue, vecAngles[idx]));
      for (int j = 0; j < max_pos_num + MATCH_CANDIDATE_NUM - 1; j++) {
        ptMaxLoc = GetNextMaxLoc(matResult, ptMaxLoc, tmpl_vecPyramid->at(top_layer).size(), value, max_overlap);
        if (value < vecLayerScores[top_layer]) {
          break;
        }
        vecMatchParams.push_back(MatchParams(Point2f(ptMaxLoc.x - fTranslationX, ptMaxLoc.y - fTranslationY), value, vecAngles[idx]));
      }
    }
    // double max_loc;
    // cv::minMaxLoc(matResult, nullptr, &max_loc, nullptr, nullptr);
    // // if (max_loc < vecLayerScores[top_layer]) {
    // // continue;
    // // }
    // std::cout << "Max loc: " << max_loc << std::endl;
  // std::string relative_path = "C:/DGB/Pick_and_place/Pc_based_software/Qt/Opencv_Console/result images/";
  // std::string save_name = relative_path + "idx_" + std::to_string(idx) + "_" + std::to_string(vecAngles[idx]) + ".bmp";
  // cv::imwrite(save_name, matRotatedSrc);
  // std::string save_name1 = relative_path + "idx_" + std::to_string(idx) + "_" + std::to_string(vecAngles[idx]) + "result.bmp";
  // cv::Mat result_write;
  // // cv::cvtColor(matResult, result_write, cv::color_) // cv::imwrite(save_name1, matResult * 255.0);
  }

  sort(vecMatchParams.begin(), vecMatchParams.end(), compareScoreBig2Small);
  std::cout << "vecMatchParams: " << vecMatchParams.size() <<std::endl;
  // console debug show
  // for (int idx=0;idx<vecMatchParams.size();idx++) {
  // std::cout << "Match params at: " << idx
  // << ", Score: " << vecMatchParams[idx]._matchScore
  // << ", Angle: " << vecMatchParams[idx]._matchAngle
  // << ", Point: " << vecMatchParams[idx]._point << std::endl;
  // }
  /// END Smallest layer matching

  int stop_layer = (stop_layer_1) ? 1 : 0;
  vector<MatchParams> vecAllResult;
  for (int i = 0; i < (int)vecMatchParams.size (); i++) {
    double rAngle = -vecMatchParams[i]._matchAngle * D2R;
    Point2f ptLT = ptRotatePt2f(vecMatchParams[i]._point, top_layerCenter, rAngle);
    double _angleStep = atan(2.0/max(top_srcWidth, top_srcHeight)) * R2D;
    vecMatchParams[i]._angleStart = vecMatchParams[i]._matchAngle - _angleStep;
    vecMatchParams[i]._angleEnd = vecMatchParams[i]._matchAngle + _angleStep;
    if (top_layer <= stop_layer) {
      vecMatchParams[i]._point = Point2d(ptLT * ((top_layer == 0) ? 1 : 2));
      vecAllResult.push_back(vecMatchParams[i]);
    } else {
      for (int iLayer = top_layer - 1; iLayer >= stop_layer; iLayer--) {
        // Search Angle
        _angleStep = atan (2.0 / max(tmpl_vecPyramid->at(iLayer).cols, tmpl_vecPyramid->at(iLayer).rows)) * R2D;
        vector<double> vecAngles;
        //double dAngleS = vecMatchParams[i].dAngleStart, dAngleE = vecMatchParams[i].dAngleEnd;
        double dMatchedAngle = vecMatchParams[i]._matchAngle;
        if (tolerance_angle < VISION_TOLERANCE) {
          vecAngles.push_back(0.0);
        } else {
          for (int i = -1; i <= 1; i++) {
            vecAngles.push_back(dMatchedAngle + _angleStep * i);
          }
        }

        Point2f ptSrcCenter((vecSrcPyr[iLayer].cols - 1) / 2.0f, (vecSrcPyr[iLayer].rows - 1) / 2.0f);
        int iSize = (int)vecAngles.size ();
        vector<MatchParams> vecNewMatchParameter(iSize);
        int iMaxScoreIndex = 0;
        double dBigValue = -1;
        for (int j = 0; j < iSize; j++) {
          Mat matResult, matRotatedSrc;
          double dMaxValue = 0;
          Point ptMaxLoc;
          GetRotatedROI(vecSrcPyr[iLayer], tmpl_vecPyramid->at(iLayer).size(), ptLT * 2, vecAngles[j], matRotatedSrc);
          // cv::imshow("Image", matRotatedSrc);
          // cv::waitKey(0);
          // std::string relative_path = "C:/DGB/Pick_and_place/Pc_based_software/Qt/Opencv_Console/result images/outroi/";
          // std::string save_name = relative_path + "roi_" + // std::to_string(i) + "_" + std::to_string(iLayer) + ".bmp";
          // cv::imwrite(save_name, matRotatedSrc);

#ifndef USE_EDGE_SIMD
          this->MatchEdgePattern(matRotatedSrc, &m_edgePattern, matResult, iLayer, vecLayerScores[iLayer], true);
#else
          this->MatchEdgePatternSIMD(matRotatedSrc, &m_edgePattern, matResult, iLayer, vecLayerScores[iLayer], true);
#endif

          minMaxLoc(matResult, 0, &dMaxValue, 0, &ptMaxLoc);
          vecNewMatchParameter[j] = MatchParams(ptMaxLoc, dMaxValue, vecAngles[j]);
          // std::cout << "Max loc: " << dMaxValue << std::endl;
          if (vecNewMatchParameter[j]._matchScore > dBigValue) {
            iMaxScoreIndex = j;
            dBigValue = vecNewMatchParameter[j]._matchScore;
          }
          // Sub-pixel estimation
          if (ptMaxLoc.x == 0 || ptMaxLoc.y == 0 || ptMaxLoc.x == matResult.cols - 1 || ptMaxLoc.y == matResult.rows - 1)
            vecNewMatchParameter[j]._posOnBorder = true;
          if (!vecNewMatchParameter[j]._posOnBorder) {
            for (int y = -1; y <= 1; y++)
              for (int x = -1; x <= 1; x++)
                vecNewMatchParameter[j]._vecResult[x + 1][y + 1] = matResult.at<float> (ptMaxLoc + Point (x, y));
          }
          // Sub-pixel estimation
        }

        if (vecNewMatchParameter[iMaxScoreIndex]._matchScore < vecLayerScores[iLayer])
          break;
        // Sub-pixel estimation
        if (sub_pixel_estimation && iLayer == 0 &&
            (!vecNewMatchParameter[iMaxScoreIndex]._posOnBorder) && iMaxScoreIndex != 0 && iMaxScoreIndex != 2) {
          double dNewX = 0, dNewY = 0, dNewAngle = 0;
          SubPixEsimation(&vecNewMatchParameter, &dNewX, &dNewY, &dNewAngle, _angleStep, iMaxScoreIndex);
          vecNewMatchParameter[iMaxScoreIndex]._point = Point2d (dNewX, dNewY);
          vecNewMatchParameter[iMaxScoreIndex]._matchAngle = dNewAngle;
        }
        // Sub-pixel estimation
        double dNewMatchAngle = vecNewMatchParameter[iMaxScoreIndex]._matchAngle;
        // Return the coordinate system to (0, 0) when it was rotated (GetRotatedROI)
        Point2f ptPaddingLT = ptRotatePt2f (ptLT * 2, ptSrcCenter, dNewMatchAngle * D2R) - Point2f (3, 3);
        Point2f pt (vecNewMatchParameter[iMaxScoreIndex]._point.x + ptPaddingLT.x,
                   vecNewMatchParameter[iMaxScoreIndex]._point.y + ptPaddingLT.y);
        // Re-rotate
        pt = ptRotatePt2f (pt, ptSrcCenter, -dNewMatchAngle * D2R);
        if (iLayer == stop_layer) {
          vecNewMatchParameter[iMaxScoreIndex]._point = pt * (stop_layer == 0 ? 1 : 2);
          vecAllResult.push_back(vecNewMatchParameter[iMaxScoreIndex]);
        } else {
          // Update MatchAngle ptLT
          vecMatchParams[i]._matchAngle = dNewMatchAngle;
          vecMatchParams[i]._angleStart = vecMatchParams[i]._matchAngle - _angleStep / 2;
          vecMatchParams[i]._angleEnd = vecMatchParams[i]._matchAngle + _angleStep / 2;
          ptLT = pt;
        }
      }
    }
  }

  FilterWithScore(&vecAllResult, min_score);
  // console debug show
  std::cout << "[Before overlap filter]" << vecAllResult.size() << std::endl;
  // for (int idx=0;idx<vecAllResult.size();idx++) {
  // std::cout << "Match params at: " << idx
  // << ", Score: " << vecAllResult[idx]._matchScore
  // << ", Angle: " << vecAllResult[idx]._matchAngle
  // << ", Point: " << vecAllResult[idx]._point << std::endl;
  // }

  // Finally, filter out the overlap
  int iDstW = tmpl_vecPyramid->at(stop_layer).cols * (stop_layer == 0 ? 1 : 2);
  int iDstH = tmpl_vecPyramid->at(stop_layer).rows * (stop_layer == 0 ? 1 : 2);
  for (int i = 0; i < (int)vecAllResult.size (); i++) {
    Point2f ptLT, ptRT, ptRB, ptLB;
    double dRAngle = -vecAllResult[i]._matchAngle * D2R;
    ptLT = vecAllResult[i]._point;
    ptRT = Point2f (ptLT.x + iDstW * (float)cos (dRAngle), ptLT.y - iDstW * (float)sin (dRAngle));
    ptLB = Point2f (ptLT.x + iDstH * (float)sin (dRAngle), ptLT.y + iDstH * (float)cos (dRAngle));
    ptRB = Point2f (ptRT.x + iDstH * (float)sin (dRAngle), ptRT.y + iDstH * (float)cos (dRAngle));
    // Record Rotation Rectangle
    vecAllResult[i]._rectR = RotatedRect(ptLT, ptRT, ptRB);
  }

  FilterWithRotatedRect(&vecAllResult, cv::TM_CCOEFF_NORMED, max_overlap);
  // Finally, filter out the overlap
  sort(vecAllResult.begin(), vecAllResult.end(), compareScoreBig2Small);
  std::cout << "[After overlap filter]" << vecAllResult.size() << std::endl;
  // for (int idx=0;idx<vecAllResult.size();idx++) {
  // std::cout << "Match params at: " << idx
  // << ", Score: " << vecAllResult[idx]._matchScore
  // << ", Angle: " << vecAllResult[idx]._matchAngle
  // << ", Point: " << vecAllResult[idx]._point << std::endl;
  // }

  matched_result.clear();
  int numOfMatched = vecAllResult.size();
  if (numOfMatched == 0) {
    return false;
  }
  int iW = tmpl_vecPyramid->at(0).cols, iH = tmpl_vecPyramid->at(0).rows;
  for (int idx=0;idx<numOfMatched;idx++) {
    MatchedObject matchObj;
    double dRAngle = -vecAllResult[idx]._matchAngle * D2R;
    matchObj.point_LT = vecAllResult[idx]._point;
    matchObj.point_RT = Point2d(matchObj.point_LT.x + iW * cos (dRAngle), matchObj.point_LT.y - iW * sin (dRAngle));
    matchObj.point_LB = Point2d(matchObj.point_LT.x + iH * sin (dRAngle), matchObj.point_LT.y + iH * cos (dRAngle));
    matchObj.point_RB = Point2d(matchObj.point_RT.x + iH * sin (dRAngle), matchObj.point_RT.y + iH * cos (dRAngle));
    matchObj.point_Center = Point2d((matchObj.point_LT.x + matchObj.point_RT.x + matchObj.point_RB.x + matchObj.point_LB.x) / 4,
                                    (matchObj.point_LT.y + matchObj.point_RT.y + matchObj.point_RB.y + matchObj.point_LB.y) / 4);
    matchObj.matched_Angle = -vecAllResult[idx]._matchAngle;
    matchObj.matched_Score = vecAllResult[idx]._matchScore;
    if (matchObj.matched_Angle < -180)
      matchObj.matched_Angle += 360;
    if (matchObj.matched_Angle > 180)
      matchObj.matched_Angle -= 360;
    matched_result.push_back(matchObj);
  }

  // debug
  std::chrono::time_point stop_point = std::chrono::high_resolution_clock::now();
  double time_count = std::chrono::duration<double, std::milli>(stop_point - start_point).count();
  std::cout << "Matching time: " << time_count << std::endl;
  cv::Mat result_image;

  cv::cvtColor(m_img_source, result_image, cv::COLOR_GRAY2BGR);

  DrawMatchResult(result_image, matched_result);
  m_matResult = result_image.clone();
  // std::string relative_path = "C:/DGB/Pick_and_place/Pc_based_software/Qt/Opencv_Console/result images/";
  // std::string save_name = relative_path + "result_image_126.bmp";
  // cv::imwrite(save_name, result_image);
  // std::cout << "Image written!" << std::endl;
  return true;
}

Size Matcher::GetBestRotationSize(Size sizeSrc, Size sizeDst, double rAngle) {
  double rAngle_radian = rAngle * D2R;
  Point ptLT (0, 0), ptLB (0, sizeSrc.height - 1), ptRB (sizeSrc.width - 1, sizeSrc.height - 1), ptRT (sizeSrc.width - 1, 0);
  Point2f ptCenter ((sizeSrc.width - 1) / 2.0f, (sizeSrc.height - 1) / 2.0f);
  Point2f ptLT_R = ptRotatePt2f (Point2f (ptLT), ptCenter, rAngle_radian);
  Point2f ptLB_R = ptRotatePt2f (Point2f (ptLB), ptCenter, rAngle_radian);
  Point2f ptRB_R = ptRotatePt2f (Point2f (ptRB), ptCenter, rAngle_radian);
  Point2f ptRT_R = ptRotatePt2f (Point2f (ptRT), ptCenter, rAngle_radian);
  float fTopY = max (max (ptLT_R.y, ptLB_R.y), max (ptRB_R.y, ptRT_R.y));
  float fBottomY = min (min (ptLT_R.y, ptLB_R.y), min (ptRB_R.y, ptRT_R.y));
  float fRightX = max (max (ptLT_R.x, ptLB_R.x), max (ptRB_R.x, ptRT_R.x));
  float fLeftX = min (min (ptLT_R.x, ptLB_R.x), min (ptRB_R.x, ptRT_R.x));

  if (rAngle > 360)
    rAngle -= 360;
  else if (rAngle < 0)
    rAngle += 360;

  if (fabs (fabs (rAngle) - 90) < VISION_TOLERANCE || fabs (fabs (rAngle) - 270) < VISION_TOLERANCE) {
    return Size (sizeSrc.height, sizeSrc.width);
  } else if (fabs (rAngle) < VISION_TOLERANCE || fabs (fabs (rAngle) - 180) < VISION_TOLERANCE) {
    return sizeSrc;
  }

  double angle = rAngle;
  if (angle > 0 && angle < 90) {
    ;
  } else if (angle > 90 && angle < 180) {
    angle -= 90;
  } else if (angle > 180 && angle < 270) {
    angle -= 180;
  } else if (angle > 270 && angle < 360) {
    angle -= 270; }
  else {
    // debug
    // "Unknown"
  }

  float fH1 = sizeDst.width * sin (angle * D2R) * cos (angle * D2R);
  float fH2 = sizeDst.height * sin (angle * D2R) * cos (angle * D2R);
  int iHalfHeight = (int)ceil (fTopY - ptCenter.y - fH1);
  int iHalfWidth = (int)ceil (fRightX - ptCenter.x - fH2);
  Size sizeRet (iHalfWidth * 2, iHalfHeight * 2);
  bool bWrongSize = (sizeDst.width < sizeRet.width && sizeDst.height > sizeRet.height) ||
                    ((sizeDst.width > sizeRet.width && sizeDst.height < sizeRet.height) ||
                    sizeDst.area () > sizeRet.area ());
  if (bWrongSize)
    sizeRet = Size (int (fRightX - fLeftX + 0.5), int (fTopY - fBottomY + 0.5));

  return sizeRet;
}

Point2f Matcher::ptRotatePt2f(Point2f ptInput, Point2f ptOrg, double angle) {
  // double dWidth = ptOrg.x * 2;
  double height = ptOrg.y * 2;
  double dY1 = height - ptInput.y;
  double dY2 = height - ptOrg.y;
  double dX = (ptInput.x - ptOrg.x) * cos (angle) - (dY1 - ptOrg.y) * sin (angle) + ptOrg.x;
  double dY = (ptInput.x - ptOrg.x) * sin (angle) + (dY1 - ptOrg.y) * cos (angle) + dY2; dY = -dY + height;
  return Point2f ((float)dX, (float)dY);
}

void Matcher::MatchEdgePattern(cv::Mat &matSrc, PatternModel *model, cv::Mat &matResult, int layer,
                               double minScores, bool fromCenter) {
  const vector<PatternModel::PatternLayer> *patterns = model->getPatterns();
  Mat imgDest;
  Mat gx;
  Mat gy;
  Mat magnitude;
  Mat angle;
  Point2f center;

  // cvtColor(matSrc, imgDest, COLOR_RGB2GRAY);
  imgDest = matSrc.clone();
  Sobel(imgDest, gx, CV_64F, 1, 0, 3);
  Sobel(imgDest, gy, CV_64F, 0, 1, 3);
  cartToPolar(gx, gy, magnitude, angle);
  vector<PatternModel::PatternPoint> modelPattern = patterns->at(layer).patternPoints;
  // ncc match search long
  long noOfCordinates = modelPattern.size();
  // normalized min score
  double normMinScore = minScores / noOfCordinates;
  double normGreediness = ((1 - m_edgeProfile.greediness * minScores) /
                           (1 - m_edgeProfile.greediness)) /
                          noOfCordinates;
  double partialScore = 0;
  matResult.create(matSrc.rows - patterns->at(layer).image.rows + 1,
                   matSrc.cols - patterns->at(layer).image.cols + 1,
                   CV_32FC1);
  matResult.setTo(0.0);
  // std::cout << matResult.size << noOfCordinates << std::endl;
  if (modelPattern.size() <= 0) {
    return;
  }
  int startRowIdx = 0;
  int endRowIdx = matResult.rows;
  int startColIdx = 0;
  int endColIdx = matResult.cols;
  if (fromCenter) {
    const int centerTolerantRow = (int)(imgDest.rows*0.3);
    const int centerTolerantCol = (int)(imgDest.cols*0.3);
    startRowIdx = modelPattern[0].Center.y - centerTolerantRow;
    endRowIdx = modelPattern[0].Center.y + centerTolerantRow;
    startColIdx = modelPattern[0].Center.x - centerTolerantCol;
    endColIdx = modelPattern[0].Center.x + centerTolerantCol;
  }
  // std::cout << endRowIdx << " " << endColIdx << std::endl;
  Point pt_center = modelPattern.at(0).Center;
  // for (int rowIdx = 0; rowIdx < matResult.rows; rowIdx++) {
  // for (int colIdx = 0; colIdx < matResult.cols; colIdx++) {
  for (int rowIdx = startRowIdx; rowIdx < endRowIdx; rowIdx++) {
    for (int colIdx = startColIdx; colIdx < endColIdx; colIdx++) {
      double partialSum = 0.0;
      // double resultScore = 0;
      for (int count = 0; count < noOfCordinates; count++) {
        PatternModel::PatternPoint tempPoint = modelPattern[count];
        int CoorX = (int)(colIdx + tempPoint.Offset.x);
        int CoorY = (int)(rowIdx + tempPoint.Offset.y);
        // ignore invalid pixel
        if (CoorX < 0 || CoorY < 0 || CoorY >(imgDest.rows - 1) || CoorX >(imgDest.cols - 1)) {
        // check invalid
        // continue;
          break;
        }
        double iTx = tempPoint.Derivative.x;
        double iTy = tempPoint.Derivative.y;
        double iSx = gx.at<double>(CoorY, CoorX);
        double iSy = gy.at<double>(CoorY, CoorX);
        if ((iSx != 0 || iSy != 0) && (iTx != 0 || iTy != 0)) {
          double mag = magnitude.at<double>(CoorY, CoorX);
          double matGradMag = (mag == 0) ? 0 : 1 / mag;
          partialSum += ((iSx * iTx) + (iSy * iTy)) * (tempPoint.Magnitude * matGradMag);
        }
        int sumOfCoords = count + 1;
        partialScore = partialSum / sumOfCoords;
        double minBreakScores = std::min((minScores - 1) + normGreediness * sumOfCoords,
                                         normMinScore * sumOfCoords);
        if (partialScore < minBreakScores) {
          // std::cout << "Break scores:" << partialScore << std::endl;
          break;
        }
      }
      if (partialScore >= minScores) {
        int tl_x = colIdx - pt_center.x;
        int tl_y = rowIdx - pt_center.y;
        if (tl_x < 0 || tl_y < 0 || tl_y > (matResult.rows - 1) || tl_x > (matResult.cols - 1)) {
          // std::cout << "Ignore scores:" << partialScore << std::endl;
          continue;
        } else {
          matResult.at<float>(tl_y, tl_x) = partialScore;
        }
      }
      // std::cout << "Partial scores:" << partialScore << std::endl;
    }
  }
}

void Matcher::MatchEdgePatternSIMD(cv::Mat &matSrc, PatternModel *model, cv::Mat &matResult,
                                   int layer, double minScores, bool fromCenter) {
  const vector<PatternModel::PatternLayer>* patterns = model->getPatterns();
  if (patterns->empty() || patterns->at(layer).patternPoints.empty())
    return;

  const auto& patternPoints = patterns->at(layer).patternPoints;
  const int patternSize = (int)patternPoints.size();
  const double normMinScore = minScores / patternSize;
  const double normGreediness = ((1 - m_edgeProfile.greediness * minScores) /
                                 (1 - m_edgeProfile.greediness)) / patternSize;

  // Convert image to grayscale float
  Mat imgGray;
  if (matSrc.channels() == 3)
    cvtColor(matSrc, imgGray, COLOR_BGR2GRAY);
  else
    imgGray = matSrc.clone();

  imgGray.convertTo(imgGray, CV_32F);
  Mat gx, gy, magnitude;
  Sobel(imgGray, gx, CV_32F, 1, 0, 3);
  Sobel(imgGray, gy, CV_32F, 0, 1, 3);
  magnitude.create(imgGray.size(), CV_32F);
  magnitude = gx.mul(gx) + gy.mul(gy);
  sqrt(magnitude, magnitude);

  int rows = matSrc.rows;
  int cols = matSrc.cols;
  const int patchH = patterns->at(layer).image.rows;
  const int patchW = patterns->at(layer).image.cols;
  const int resultH = rows - patchH + 1;
  const int resultW = cols - patchW + 1;

  matResult.create(resultH, resultW, CV_32FC1);
  matResult.setTo(0.0f);

  float* gxData = (float*)gx.data;
  float* gyData = (float*)gy.data;
  float* magData = (float*)magnitude.data;
  const int imgStride = gx.cols;

  int startRowIdx = 0, endRowIdx = resultH;
  int startColIdx = 0, endColIdx = resultW;
  Point pt_center = patternPoints[0].Center;

  if (fromCenter) {
    const int centerTolerantRow = (int)(imgGray.rows * 0.3);
    const int centerTolerantCol = (int)(imgGray.cols * 0.3);
    // startRowIdx = max(0, patternPoints[0].Center.y - centerTolerantRow);
    // endRowIdx = min(resultH, patternPoints[0].Center.y + centerTolerantRow);
    // startColIdx = max(0, patternPoints[0].Center.x - centerTolerantCol);
    // endColIdx = min(resultW, patternPoints[0].Center.x + centerTolerantCol);
    startRowIdx = patternPoints[0].Center.y - centerTolerantRow;
    endRowIdx = patternPoints[0].Center.y + centerTolerantRow;
    startColIdx = patternPoints[0].Center.x - centerTolerantCol;
    endColIdx = patternPoints[0].Center.x + centerTolerantCol;
  }

#pragma omp parallel for
  for (int rowIdx = startRowIdx; rowIdx < endRowIdx; ++rowIdx) {
    for (int colIdx = startColIdx; colIdx < endColIdx; ++colIdx) {
      __m256 sum = _mm256_setzero_ps();
      float totalSum = 0.0f;
      int count = 0;

      for (; count + 8 <= patternSize; count += 8) {
        __m256 iSx, iSy, iTx, iTy, magVal, invMag, magWeight;

        float sx[8], sy[8], mag8[8], tx[8], ty[8], mw[8];
        bool valid = true;

        for (int i = 0; i < 8; ++i) {
          int x = colIdx + patternPoints[count + i].Offset.x;
          int y = rowIdx + patternPoints[count + i].Offset.y;

          if (x < 0 || y < 0 || x >= cols || y >= rows) {
            valid = false;
            break;
          }

          int idx = y * imgStride + x;
          sx[i] = gxData[idx];
          sy[i] = gyData[idx];
          mag8[i] = magData[idx];
          tx[i] = patternPoints[count + i].Derivative.x;
          ty[i] = patternPoints[count + i].Derivative.y;
          mw[i] = patternPoints[count + i].Magnitude;
        }

        if (!valid)
          break;

        iSx = _mm256_loadu_ps(sx);
        iSy = _mm256_loadu_ps(sy);
        iTx = _mm256_loadu_ps(tx);
        iTy = _mm256_loadu_ps(ty);
        magVal = _mm256_loadu_ps(mag8);
        magWeight = _mm256_loadu_ps(mw);

        __m256 dot = _mm256_add_ps(_mm256_mul_ps(iSx, iTx), _mm256_mul_ps(iSy, iTy));
        invMag = _mm256_rcp_ps(magVal); // approximate 1/mag
        dot = _mm256_mul_ps(dot, _mm256_mul_ps(magWeight, invMag));
        sum = _mm256_add_ps(sum, dot);
      }

      float buffer[8];
      _mm256_storeu_ps(buffer, sum);
      for (int i = 0; i < 8; ++i)
        totalSum += buffer[i];

      // Handle remaining points
      for (; count < patternSize; ++count) {
        int x = colIdx + patternPoints[count].Offset.x;
        int y = rowIdx + patternPoints[count].Offset.y;
        if (x < 0 || y < 0 || x >= cols || y >= rows)
          break;
        int idx = y * imgStride + x;
        float sx = gxData[idx];
        float sy = gyData[idx];
        float mag = magData[idx];
        if (mag == 0.0f) continue;

        float tx = patternPoints[count].Derivative.x;
        float ty = patternPoints[count].Derivative.y;
        float mw = patternPoints[count].Magnitude;
        totalSum += ((sx * tx + sy * ty) * (mw / mag));

        int num = count + 1;
        float partialScore = totalSum / num;
        float minBreak = std::min((float)((minScores - 1) + normGreediness * num),
                                  (float)(normMinScore * num));
        if (partialScore < minBreak)
          break;
      }

      float finalScore = totalSum / patternSize;
      if (finalScore >= minScores) {
        int tl_x = colIdx - pt_center.x;
        int tl_y = rowIdx - pt_center.y;
        if (tl_x >= 0 && tl_y >= 0 && tl_x < resultW && tl_y < resultH)
          matResult.at<float>(tl_y, tl_x) = finalScore;
      }
    }
  }
}

void Matcher::MatchPattern(cv::Mat &matSrc, Tmpl* pTmplData, cv::Mat &matResult,
                           int layer, bool useSIMD) {
  if (useSIMD) {
    matResult.create (matSrc.rows - pTmplData->getPyramid()->at(layer).rows + 1,
                     matSrc.cols - pTmplData->getPyramid()->at(layer).cols + 1, CV_32FC1);
    matResult.setTo (0);
    cv::Mat matTemplate = pTmplData->getPyramid()->at(layer).clone();
    int t_r_end = matTemplate.rows, t_r = 0;
    for (int r = 0; r < matResult.rows; r++) {
      float* r_matResult = matResult.ptr<float> (r);
      uchar* r_source = matSrc.ptr<uchar> (r);
      uchar* r_template, *r_sub_source;
      for (int c = 0; c < matResult.cols; ++c, ++r_matResult, ++r_source) {
        r_template = matTemplate.ptr<uchar> ();
        r_sub_source = r_source;
        for (t_r = 0; t_r < t_r_end; ++t_r, r_sub_source += matSrc.cols, r_template += matTemplate.cols) {
          *r_matResult = *r_matResult + IM_Conv_SIMD (r_template, r_sub_source, matTemplate.cols);
        }
      }
    }
  } else {
    cv::Mat templ_image = pTmplData->getPyramid()->at(layer).clone();
    cv::matchTemplate(matSrc, templ_image, matResult, cv::TM_CCORR);
  }

  CCOEFF_Denominator(matSrc, pTmplData, matResult, layer);
}

void Matcher::CCOEFF_Denominator(cv::Mat &matSrc, Tmpl *pTmplData,
                                 cv::Mat &matResult, int layer) {
  if (pTmplData->getResultEqual1()->at(layer)) {
    matResult = Scalar::all (1);
    return;
  }
  double *q0 = 0, *q1 = 0, *q2 = 0, *q3 = 0;
  const vector<Mat>* vecPyramid = pTmplData->getPyramid();
  const vector<Scalar>* vecTemplMean = pTmplData->getTemplMean();
  const vector<double>* vecTemplNorm = pTmplData->getTemplNorm();
  const vector<double>* vecInvArea = pTmplData->getInvArea();
  Mat sum, sqsum;
  integral (matSrc, sum, sqsum, CV_64F);
  q0 = (double*)sqsum.data;
  q1 = q0 + vecPyramid->at(layer).cols;
  q2 = (double*)(sqsum.data + vecPyramid->at(layer).rows * sqsum.step);
  q3 = q2 + vecPyramid->at(layer).cols;
  double* p0 = (double*)sum.data;
  double* p1 = p0 + vecPyramid->at(layer).cols;
  double* p2 = (double*)(sum.data + vecPyramid->at(layer).rows*sum.step);
  double* p3 = p2 + vecPyramid->at(layer).cols;
  int sumstep = sum.data ? (int)(sum.step / sizeof (double)) : 0;
  int sqstep = sqsum.data ? (int)(sqsum.step / sizeof (double)) : 0;

  double dTemplMean0 = vecTemplMean->at(layer)[0];
  double dTemplNorm = vecTemplNorm->at(layer);
  double dInvArea = vecInvArea->at(layer);


  int i, j;
  for (i = 0; i < matResult.rows; i++) {
    float* rrow = matResult.ptr<float> (i);
    int idx = i * sumstep;
    int idx2 = i * sqstep;
    for (j = 0; j < matResult.cols; j += 1, idx += 1, idx2 += 1) {
      double num = rrow[j], t;
      double wndMean2 = 0, wndSum2 = 0; t = p0[idx] - p1[idx] - p2[idx] + p3[idx];
      wndMean2 += t * t; num -= t * dTemplMean0;
      wndMean2 *= dInvArea; t = q0[idx2] - q1[idx2] - q2[idx2] + q3[idx2];
      wndSum2 += t;
      //t = std::sqrt (MAX (wndSum2 - wndMean2, 0)) * dTemplNorm;
      double diff2 = MAX (wndSum2 - wndMean2, 0);
      if (diff2 <= std::min (0.5, 10 * FLT_EPSILON * wndSum2))
        t = 0; // avoid rounding errors
      else
        t = std::sqrt (diff2)*dTemplNorm;

      if (fabs (num) < t)
        num /= t;
      else if (fabs (num) < t * 1.125)
        num = num > 0 ? 1 : -1;
      else
        num = 0;

      rrow[j] = (float)num;
    }
  }
}

Point Matcher::GetNextMaxLoc (Mat & matResult, Point ptMaxLoc,
                             Size sizeTemplate, double& dMaxValue,
                             double dMaxOverlap) {
  //比對到的區域需考慮重疊比例
  int iStartX = ptMaxLoc.x - sizeTemplate.width * (1 - dMaxOverlap);
  int iStartY = ptMaxLoc.y - sizeTemplate.height * (1 - dMaxOverlap);
  //塗黑
  rectangle(matResult, Rect(iStartX,
                            iStartY, 2 * sizeTemplate.width * (1- dMaxOverlap),
                            2 * sizeTemplate.height * (1- dMaxOverlap)),
                            Scalar (-1), cv::FILLED);
  //得到下一個最大值
  Point ptNewMaxLoc;
  minMaxLoc (matResult, 0, &dMaxValue, 0, &ptNewMaxLoc);
  return ptNewMaxLoc;
}

Point Matcher::GetNextMaxLoc(Mat &matResult, Point ptMaxLoc,
                             Size sizeTemplate, double &maxValue,
                             double maxOverlap, BlockMax &blockMax) {
  // The overlap ratio of the matched areas should be considered
  int iStartX = int (ptMaxLoc.x - sizeTemplate.width * (1 - maxOverlap));
  int iStartY = int (ptMaxLoc.y - sizeTemplate.height * (1 - maxOverlap));
  Rect rectIgnore (iStartX, iStartY, int (2 * sizeTemplate.width * (1 - maxOverlap)) ,
                  int (2 * sizeTemplate.height * (1 - maxOverlap)));
  // Blackened
  cv::rectangle(matResult, rectIgnore , Scalar (-1), cv::FILLED);
  blockMax.UpdateMax(rectIgnore);
  Point ptReturn;
  blockMax.GetMaxValueLoc(maxValue, ptReturn);
  return ptReturn;
}

void Matcher::GetRotatedROI(Mat &matSrc, Size size, Point2f ptLT, double dAngle, Mat &matROI) {
  double dAngle_radian = dAngle * D2R;
  Point2f ptC ((matSrc.cols - 1) / 2.0f, (matSrc.rows - 1) / 2.0f);
  Point2f ptLT_rotate = ptRotatePt2f (ptLT, ptC, dAngle_radian);
  Size sizePadding (size.width + 6, size.height + 6);
  Mat rMat = getRotationMatrix2D (ptC, dAngle, 1);
  rMat.at<double> (0, 2) -= ptLT_rotate.x - 3;
  rMat.at<double> (1, 2) -= ptLT_rotate.y - 3;

  //平移旋轉矩陣(0, 2) (1, 2)的減，為旋轉後的圖形偏移，-= ptLT_rotate.x - 3 代表旋轉後的圖形往-X方向移動ptLT_rotate.x - 3
  warpAffine (matSrc, matROI, rMat, sizePadding);
}

bool Matcher::SubPixEsimation(vector<MatchParams> *vec, double *dNewX, double *dNewY, double *dNewAngle,
                              double dAngleStep, int iMaxScoreIndex) {
  //Az=S, (A.T)Az=(A.T)s, z = ((A.T)A).inv (A.T)s
  Mat matA (27, 10, CV_64F);
  Mat matZ (10, 1, CV_64F);
  Mat matS (27, 1, CV_64F);
  double dX_maxScore = (*vec)[iMaxScoreIndex]._point.x;
  double dY_maxScore = (*vec)[iMaxScoreIndex]._point.y;
  double dTheata_maxScore = (*vec)[iMaxScoreIndex]._matchAngle;
  int iRow = 0;

  /*for (int x = -1; x <= 1; x++) {
      for (int y = -1; y <= 1; y++) {
        for (int theta = 0; theta <= 2; theta++) {*/
  for (int theta = 0; theta <= 2; theta++) {
    for (int y = -1; y <= 1; y++) {
      for (int x = -1; x <= 1; x++) {
        //xx yy tt xy xt yt x y t 1
        //0 1 2 3 4 5 6 7 8 9
        double dX = dX_maxScore + x;
        double dY = dY_maxScore + y;
        //double dT = (*vec)[theta].dMatchAngle + (theta - 1) * dAngleStep;
        double dT = (dTheata_maxScore + (theta - 1) * dAngleStep) * D2R;
        matA.at<double> (iRow, 0) = dX * dX;
        matA.at<double> (iRow, 1) = dY * dY;
        matA.at<double> (iRow, 2) = dT * dT;
        matA.at<double> (iRow, 3) = dX * dY;
        matA.at<double> (iRow, 4) = dX * dT;
        matA.at<double> (iRow, 5) = dY * dT;
        matA.at<double> (iRow, 6) = dX;
        matA.at<double> (iRow, 7) = dY;
        matA.at<double> (iRow, 8) = dT;
        matA.at<double> (iRow, 9) = 1.0;
        matS.at<double> (iRow, 0) = (*vec)[iMaxScoreIndex + (theta - 1)]._vecResult[x + 1][y + 1];
        iRow++;

    #ifdef _DEBUG
      // string str = format ("%.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f",
      //                       dValueA[0], dValueA[1], dValueA[2], dValueA[3], dValueA[4], dValueA[5],
      //                       dValueA[6], dValueA[7], dValueA[8], dValueA[9]); fileA << str << endl;
      // str = format ("%.6f", dValueS[iRow]); fileS << str << endl;
    #endif
      }
    }
  }

  //求解Z矩陣，得到k0~k9
  //[ x* ] = [ 2k0 k3 k4 ]-1 [ -k6 ]
  //| y* | = | k3 2k1 k5 | | -k7 |
  //[ t* ] = [ k4 k5 2k2 ] [ -k8 ]

  //solve (matA, matS, matZ, DECOMP_SVD);
  matZ = (matA.t () * matA).inv () * matA.t ()* matS;
  Mat matZ_t;
  transpose (matZ, matZ_t);
  double* dZ = matZ_t.ptr<double> (0);
  Mat matK1 = (Mat_<double> (3, 3) << (2 * dZ[0]), dZ[3], dZ[4], dZ[3], (2 * dZ[1]), dZ[5], dZ[4], dZ[5], (2 * dZ[2]));
  Mat matK2 = (Mat_<double> (3, 1) << -dZ[6], -dZ[7], -dZ[8]);
  Mat matDelta = matK1.inv () * matK2;
  *dNewX = matDelta.at<double> (0, 0);
  *dNewY = matDelta.at<double> (1, 0);
  *dNewAngle = matDelta.at<double> (2, 0) * R2D;
  return true;
}

void Matcher::FilterWithScore(vector<MatchParams>* vec, double dScore) {
  sort(vec->begin(), vec->end(), compareScoreBig2Small);
  int iSize = vec->size (), iIndexDelete = iSize + 1;
  for (int i = 0; i < iSize; i++) {
    if ((*vec)[i]._matchScore < dScore) {
      iIndexDelete = i;
      break;
    }
  }
  if (iIndexDelete == iSize + 1) //沒有任何元素小於dScore
    return;
  vec->erase(vec->begin() + iIndexDelete, vec->end());
  return;
}

void Matcher::FilterWithRotatedRect(vector<MatchParams> *vec, int iMethod, double dMaxOverLap) {
  int iMatchSize = (int)vec->size ();
  RotatedRect rect1, rect2;
  for (int i = 0; i < iMatchSize - 1; i++) {
    if (vec->at (i)._delete)
      continue;
    for (int j = i + 1; j < iMatchSize; j++) {
      if (vec->at (j)._delete)
        continue;
      rect1 = vec->at (i)._rectR; rect2 = vec->at (j)._rectR;
      vector<Point2f> vecInterSec;
      int iInterSecType = rotatedRectangleIntersection (rect1, rect2, vecInterSec);
      if (iInterSecType == cv::INTERSECT_NONE) //無交集
        continue;
      else if (iInterSecType == cv::INTERSECT_FULL) { //一個矩形包覆另一個
        int iDeleteIndex;
        if (iMethod == cv::TM_SQDIFF)
          iDeleteIndex = (vec->at (i)._matchScore <= vec->at (j)._matchScore) ? j : i;
        else
          iDeleteIndex = (vec->at (i)._matchScore >= vec->at (j)._matchScore) ? j : i;
        vec->at (iDeleteIndex)._delete = true;
      } else { //交點 > 0
        if (vecInterSec.size () < 3) //一個或兩個交點
          continue;
        else {
          int iDeleteIndex; //求面積與交疊比例
          SortPtWithCenter(vecInterSec);
          double dArea = contourArea (vecInterSec);
          double dRatio = dArea / rect1.size.area (); //若大於最大交疊比例，選分數高的
          if (dRatio > dMaxOverLap) {
            if (iMethod == cv::TM_SQDIFF)
              iDeleteIndex = (vec->at (i)._matchScore <= vec->at (j)._matchScore) ? j : i;
            else
              iDeleteIndex = (vec->at (i)._matchScore >= vec->at (j)._matchScore) ? j : i;
            vec->at (iDeleteIndex)._delete = true;
          }
        }
      }
    }
  }

  vector<MatchParams>::iterator it;
  for (it = vec->begin (); it != vec->end ();) {
    if ((*it)._delete)
      it = vec->erase (it);
    else ++it;
  }
}

void Matcher::SortPtWithCenter(vector<Point2f>& vecSort) {
  int iSize = (int)vecSort.size ();
  Point2f ptCenter;
  for (int i = 0; i < iSize; i++)
    ptCenter += vecSort[i];

  ptCenter /= iSize;
  Point2f vecX (1, 0);
  vector<pair<Point2f, double>> vecPtAngle (iSize);
  for (int i = 0; i < iSize; i++) {
    vecPtAngle[i].first = vecSort[i]; //pt
    Point2f vec1 (vecSort[i].x - ptCenter.x, vecSort[i].y - ptCenter.y);
    float fNormVec1 = vec1.x * vec1.x + vec1.y * vec1.y;
    float fDot = vec1.x; if (vec1.y < 0) { //若點在中心的上方
      vecPtAngle[i].second = acos (fDot / fNormVec1) * R2D;
    } else if (vec1.y > 0) { //下方
      vecPtAngle[i].second = 360 - acos (fDot / fNormVec1) * R2D;
    } else { //點與中心在相同Y
      if (vec1.x - ptCenter.x > 0)
        vecPtAngle[i].second = 0;
      else vecPtAngle[i].second = 180;
    }
  }
  sort (vecPtAngle.begin (), vecPtAngle.end (), comparePtWithAngle);
  for (int i = 0; i < iSize; i++) vecSort[i] = vecPtAngle[i].first;
}

void Matcher::DrawMatchResult(Mat &drawImage, vector<MatchedObject> &matchedResults) {
  cv::Scalar box_color(255, 0, 0);
  for(int index=0;index<matchedResults.size();index++) {
    MatchedObject *obj = &matchedResults[index];
    cv::line(drawImage, obj->point_LT, obj->point_RT, box_color, 2, cv::LINE_AA);
    cv::line(drawImage, obj->point_RT, obj->point_RB, box_color, 2, cv::LINE_AA);
    cv::line(drawImage, obj->point_RB, obj->point_LB, box_color, 2, cv::LINE_AA);
    cv::line(drawImage, obj->point_LB, obj->point_LT, box_color, 2, cv::LINE_AA);
  }
}
