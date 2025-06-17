#ifndef MATCHER_H
#define MATCHER_H

#include "tmpl.h"
#include "blockmax.h"
#include "matchparams.h"
#include "matchedobject.h"
#include "patternmodel.h"

#define VISION_TOLERANCE 0.0000001
#define D2R (CV_PI / 180.0)
#define R2D (180.0 / CV_PI)
#define MATCH_CANDIDATE_NUM 5

// #define USE_EDGE_SIMD

class Matcher {
  public:
    Matcher();
    void setEdgePatternImage(cv::Mat &img_pattern);
    void setPatternImage(cv::Mat &img_pattern);
    void setMatchSourceImage(cv::Mat &img);
    bool Match();
    bool MatchEdge();
    // bool MatchEdgeImprove();

  private:
    void MatchEdgePattern(cv::Mat &matSrc, PatternModel *model,
                          cv::Mat &matResult, int layer, double minScores,
                          bool fromCenter=false);
    void MatchEdgePatternSIMD(cv::Mat &matSrc, PatternModel *model,
                          cv::Mat &matResult, int layer, double minScores,
                          bool fromCenter=false);
    // void MatchEdgePattern_SIMD(cv::Mat &matSrc, PatternModel *model,
    //                            cv::Mat &matResult, int layer,
    //                            double minScores, bool fromCenter=false);
    void MatchPattern(cv::Mat &matSrc, Tmpl *pTmplData,
                      cv::Mat &matResult, int layer, bool useSIMD);
    void CCOEFF_Denominator(cv::Mat &matSrc, Tmpl *pTmplData,
                            cv::Mat &matResult, int layer);
    Size GetBestRotationSize(Size sizeSrc, Size sizeDst, double rAngle);
    Point2f ptRotatePt2f(Point2f ptInput, Point2f ptOrg, double angle);
    Point GetNextMaxLoc(Mat &matResult, Point ptMaxLoc, Size sizeTemplate,
                        double& maxValue, double maxOverlap);
    Point GetNextMaxLoc(Mat & matResult, Point ptMaxLoc, Size sizeTemplate,
                        double & maxValue, double maxOverlap,
                        BlockMax &blockMax);
    void GetRotatedROI(Mat &matSrc, Size size, Point2f ptLT,
                       double dAngle, Mat &matROI);
    bool SubPixEsimation(vector<MatchParams> *vec, double *dNewX,
                         double *dNewY, double *dNewAngle,
                         double dAngleStep, int iMaxScoreIndex);
    void FilterWithScore(vector<MatchParams>* vec, double dScore);
    void FilterWithRotatedRect(vector<MatchParams> *vec, int iMethod,
                               double dMaxOverLap);
    void SortPtWithCenter(vector<Point2f>& vecSort);
    void DrawMatchResult(Mat &drawImage,
                         vector<MatchedObject> &matchedResults);

  public:
    int max_pos_num = 70;
    double max_overlap = 0.2;
    int min_reduce_length = 8;
    double tolerance_angle = 180;
    double min_score = 0.85;
    bool sub_pixel_estimation = false;
    bool stop_layer_1 = false;
    vector<MatchedObject> matched_result;
    cv::Mat m_matResult;

  private:
    cv::Mat m_img_source;
    Tmpl m_pattern;
    EdgeProfile m_edgeProfile;
    PatternModel m_edgePattern;
};

#endif // MATCHER_H
