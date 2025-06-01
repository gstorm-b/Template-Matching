#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "matching/matcher.h"
// #include "matching/template_matching.h"

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui_c.h>

using namespace cv;

QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}
QT_END_NAMESPACE

class MainWindow : public QMainWindow {
  Q_OBJECT

public:
  MainWindow(QWidget *parent = nullptr);
  ~MainWindow();

private:
  void btn_match_clicked();

  bool qpixmap2Mat(QPixmap &pixmap, cv::Mat &outmat);
  bool mat2Qpixmap(cv::Mat &inmat, QPixmap &pixmap);

private:
  Ui::MainWindow *ui;
  // TmplMatching matcher;
};
#endif // MAINWINDOW_H
