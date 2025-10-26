#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(
    QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow) {
  ui->setupUi(this);

  connect(ui->btn_matching, &QPushButton::clicked,
          this, &MainWindow::btn_match_clicked);

}

MainWindow::~MainWindow() {
  delete ui;
}

void MainWindow::btn_match_clicked() {

  cv::Mat img_source;
  QPixmap pixmap_source = ui->grpview_source->getImage();
  qpixmap2Mat(pixmap_source, img_source);
  // qDebug() << "Source image size:" << img_source.cols << img_source.rows << img_source.type();

  cv::Mat img_dst;
  QPixmap pixmap_dst = ui->grpview_dst->getImage();
  qpixmap2Mat(pixmap_dst, img_dst);
  // qDebug() << "Destination image size:" << img_dst.cols << img_dst.rows << img_dst.type();

  Matcher matcher;
  matcher.max_pos_num = 50;
  matcher.max_overlap = 0.2;
  matcher.min_reduce_length = 32;
  matcher.tolerance_angle = 180.0;
  matcher.min_score = 0.9;
  matcher.sub_pixel_estimation = false;
  matcher.stop_layer_1 = false;

  cv::cvtColor(img_dst, img_dst, cv::COLOR_BGR2GRAY);
  cv::cvtColor(img_source, img_source, cv::COLOR_BGR2GRAY);
  matcher.setPatternImage(img_dst);
  matcher.setEdgePatternImage(img_dst);
  matcher.setMatchSourceImage(img_source);

  // bool match_result = matcher.Match();
  bool match_result = matcher.MatchEdge();

  if (match_result) {
    QPixmap result_img;
    cv::Mat mat_result_img = matcher.m_matResult.clone();
    if (mat2Qpixmap(mat_result_img, result_img)) {
      ui->grpview_show->loadImage(result_img);
    }
  }
}

bool MainWindow::qpixmap2Mat(QPixmap &pixmap, cv::Mat &outmat) {
  QImage image = pixmap.toImage();
  switch (image.format()) {
    case QImage::Format_Grayscale8:
      outmat = cv::Mat(image.height(), image.width(),
                       CV_8UC1, (void*)image.constBits(), image.bytesPerLine());
      break;
    case QImage::Format_RGB32:
      outmat = cv::Mat(image.height(), image.width(),
                       CV_8UC4, (void*)image.constBits(), image.bytesPerLine());
      cv::cvtColor(outmat, outmat, cv::COLOR_BGRA2BGR);
      break;
    case QImage::Format_ARGB32:
      outmat = cv::Mat(image.height(), image.width(),
                       CV_8UC4, (void*)image.constBits(), image.bytesPerLine());
      break;
    default:
      qWarning("qpixmap2Mat: Unknown type!");
      return false;
  }
  return true;
}

bool MainWindow::mat2Qpixmap(cv::Mat &inmat, QPixmap &pixmap) {
  QImage img;

  switch (inmat.type()) {
    case CV_8UC1:
      img = QImage(inmat.data, inmat.cols, inmat.rows,
                   static_cast<int>(inmat.step), QImage::Format_Grayscale8);
      break;
    case CV_8UC3:
        img = QImage(inmat.data, inmat.cols, inmat.rows,
                     static_cast<int>(inmat.step), QImage::Format_RGB888);
      break;
    case CV_8UC4:
        img = QImage(inmat.data, inmat.cols, inmat.rows,
                     static_cast<int>(inmat.step), QImage::Format_ARGB32);
      break;
    default:
      qWarning("mat2Qpixmap: Unknown type!");
      return false;
  }

  QImage imgCopy = img.copy();
  pixmap = QPixmap::fromImage(imgCopy);
  return true;
}
