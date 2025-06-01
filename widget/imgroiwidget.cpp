#include "imgroiwidget.h"

#include <QFileDialog>
#include <QMenu>

#include <QDebug>
#include <QGraphicsPixmapItem>
#include <QScrollBar>

ImgRoiWidget::ImgRoiWidget(QWidget *parent)
    : QGraphicsView(parent),
    m_scene(new QGraphicsScene(this)),
    m_pixmapItem(nullptr),
    m_setting(nullptr),
    m_using_user_setting(false),
    m_current_mode(IModeNone),
    m_previous_mode(IModeNone),
    m_scene_interacting(false),
    m_has_panned(false),
    m_roi_started(false) {

  this->setScene(m_scene);
  // m_roi_group = new QGraphicsItemGroup(this);
  // m_scene->addItem(m_roi_group);

  QBrush brush(QColor(0x222222));
  this->setBackgroundBrush(brush);

  // set view port update mode to avoid ghosting
  this->setViewportUpdateMode(QGraphicsView::FullViewportUpdate);
  // off smooth pixmap transform to disable blur when zoom in
  this->setRenderHint(QPainter::SmoothPixmapTransform, false);

  init_mouse_menu();
}

void ImgRoiWidget::setSettings(QSettings *setting) {
  if (setting == nullptr) {
    return;
  }

  if ((!m_using_user_setting) && (this->m_setting != nullptr)){
    delete this->m_setting;
  }

  this->m_setting = setting;
  m_using_user_setting = true;
}

void ImgRoiWidget::removeSettings() {
  if ((m_using_user_setting) && (this->m_setting != nullptr)){
    this->m_setting = nullptr;
    m_using_user_setting = false;
  }
}

QPixmap ImgRoiWidget::getImage() {
  return m_pixmapItem->pixmap();
}

void ImgRoiWidget::loadImage(const QString &filePath) {
  QImage image(filePath);
  if (image.isNull()) {
    qDebug() << "Failed to load image:" << filePath;
    return;
  }

  QPixmap pixmap = QPixmap::fromImage(image);
  if (!m_pixmapItem) {
    m_pixmapItem = m_scene->addPixmap(pixmap);
    // using FastTransformation to avoid blur image
    m_pixmapItem->setTransformationMode(Qt::FastTransformation);
  } else {
    m_pixmapItem->setPixmap(pixmap);
  }
  m_scene->setSceneRect(pixmap.rect());
  // fit image with current window size
  m_pixmap_bounding_rect = m_pixmapItem->boundingRect();
  this->fitInView(m_pixmap_bounding_rect, Qt::KeepAspectRatio);
}

void ImgRoiWidget::loadImage(QPixmap &pixmap) {
  if (!m_pixmapItem) {
    m_pixmapItem = m_scene->addPixmap(pixmap);
    // using FastTransformation to avoid blur image
    m_pixmapItem->setTransformationMode(Qt::FastTransformation);
  } else {
    m_pixmapItem->setPixmap(pixmap);
  }
  m_scene->setSceneRect(pixmap.rect());
  // fit image with current window size
  m_pixmap_bounding_rect = m_pixmapItem->boundingRect();
  this->fitInView(m_pixmap_bounding_rect, Qt::KeepAspectRatio);
}

void ImgRoiWidget::removeImage() {
  if (!hadImage()) {
    qDebug() << "[IMG ROI Widget] Remove failed, image empty.";
    return;
  }

  m_scene->removeItem(m_pixmapItem);
  delete m_pixmapItem;
  m_pixmapItem = nullptr;
  this->fitInView(m_pixmap_bounding_rect, Qt::KeepAspectRatio);
}

void ImgRoiWidget::startDrawROI() {
  if (m_current_mode == IModeNone) {
    // this->setCursor(Qt::CrossCursor);
    m_scene->clearSelection();
    changeInteractMode(IModeDrawROI);
  }
}

void ImgRoiWidget::deletedSelectedItems() {
  QList<QGraphicsItem*> selected_items = m_scene->selectedItems();
  if (selected_items.empty()) {
    return;
  }

  for (int idx=0;idx<selected_items.count();idx++) {
    QGraphicsItem *item = selected_items[idx];
    m_scene->removeItem(item);
    delete item;
  }
}

void ImgRoiWidget::mousePressEvent(QMouseEvent *event) {
  // custom handle mouse press event
  switch (event->button()) {
    case Qt::RightButton:
      if (this->rightMouseButtonPressed(event)) {
        return;
      }
      break;
    case Qt::LeftButton:
      if (this->leftMouseButtonPressed(event)) {
        return;
      }
      break;
    default:
      break;
  }

  QGraphicsView::mousePressEvent(event);
}

void ImgRoiWidget::mouseMoveEvent(QMouseEvent *event) {

  switch (m_current_mode) {
    case IModeNone:

      break;
    case IModeZoom:

      break;
    case IModePan:
      {
        m_has_panned = true;
        QPoint delta = event->pos() - m_last_pan_point;
        if (!delta.isNull()) {
          this->horizontalScrollBar()->setValue(
              horizontalScrollBar()->value() - delta.x());
          this->verticalScrollBar()->setValue(
              verticalScrollBar()->value() - delta.y());
          m_last_pan_point = event->pos();
        }
      }
      break;
    case IModeDrawROI:
      {

        QPointF currentPos = mapToScene(event->pos());
        QRectF newRect(m_roi_start_point, currentPos);

        newRect = newRect.normalized();
        m_temp_roi->setRect(newRect);
        qDebug() << "[IMG ROI Widget] Add new ROI: change position." << newRect;
        return;
      }
      break;
  }

  QGraphicsView::mouseMoveEvent(event);
}

void ImgRoiWidget::mouseReleaseEvent(QMouseEvent *event) {
  // custom handle mouse release event
  switch (event->button()) {
    case Qt::RightButton:
      if (this->rightMouseButtonReleased(event)) {
        return;
      }
      break;
    case Qt::LeftButton:
      if (this->leftMouseButtonReleased(event)) {
        return;
      }
      break;
    default:
      break;
  }

  QGraphicsView::mouseReleaseEvent(event);
}

void ImgRoiWidget::mouseDoubleClickEvent(QMouseEvent *event) {

  QGraphicsView::mouseDoubleClickEvent(event);
}

void ImgRoiWidget::wheelEvent(QWheelEvent *event) {
  if (event->modifiers() & Qt::ControlModifier) {
    changeInteractMode(IModeZoom);
    double angle = event->angleDelta().y();
    double factor = (angle > 0) ? 1.15 : 0.85;
    this->scale(factor, factor);
    backToPreviousMode();
    return;
  }

  QGraphicsView::wheelEvent(event);
}

void ImgRoiWidget::keyPressEvent(QKeyEvent *event) {
  if (event->key() == Qt::Key_Escape) {
    changeInteractMode(IModeNone);
    event->accept();
  }

  if (m_current_mode == IModeNone) {
    switch (event->key()) {
      case Qt::Key_Delete:
        deletedSelectedItems();
        event->accept();
        break;

      case Qt::Key_A:
        startDrawROI();
        event->accept();
        break;

      default:
        break;
    }
  }

  QGraphicsView::keyPressEvent(event);
}

void ImgRoiWidget::keyReleaseEvent(QKeyEvent *event) {
  if ((event->key() == Qt::Key_Control) && (m_current_mode == IModePan)) {
    m_last_pan_point = QPoint();
    // this->setCursor(Qt::ArrowCursor);
    changeInteractMode(IModeNone);
    event->accept();
    // return;
  }

  QGraphicsView::keyReleaseEvent(event);
}

void ImgRoiWidget::changeInteractMode(InteractMode mode) {
  if (mode != m_current_mode) {
    m_previous_mode = m_current_mode;
    m_current_mode = mode;
    changeCursor();
    m_scene_interacting = (m_current_mode != IModeNone) ? true : false;
    // qDebug() << "[IMG ROI Widget] Chang mode:"
    //          << interactMode2String(m_previous_mode)
    //          << ">>"
    //          << interactMode2String(m_current_mode);
  }
}

void ImgRoiWidget::backToPreviousMode() {
  if (m_previous_mode == m_current_mode) {
    return;
  }

  InteractMode temp_mode = m_current_mode;
  m_current_mode = m_previous_mode;
  m_previous_mode = temp_mode;
  changeCursor();
  m_scene_interacting = (m_current_mode != IModeNone) ? true : false;

  // qDebug() << "[IMG ROI Widget] Chang mode (back):"
  //          << interactMode2String(m_previous_mode)
  //          << ">>"
  //          << interactMode2String(m_current_mode);
}

void ImgRoiWidget::changeCursor() {
  switch (m_current_mode) {
    case IModeNone:
      this->setCursor(Qt::ArrowCursor);
      break;
    case IModeZoom:

      break;
    case IModePan:
      this->setCursor(Qt::ClosedHandCursor);
      break;
    case IModeDrawROI:
      this->setCursor(Qt::CrossCursor);
      break;
  }
}

void ImgRoiWidget::init_mouse_menu() {
  action_add_roi = new QAction("Add ROI", this);
  action_delete_roi = new QAction("Delete selected ROIs", this);
  action_save_roi = new QAction("Save ROIs", this);
  action_load_img = new QAction("Load Image", this);
  action_remove_img = new QAction("Remove Image", this);
  action_reset_img = new QAction("Reset transform", this);

  menu_right_mouse = new QMenu();
  menu_right_mouse->addAction(action_load_img);

  menu_right_mouse_full = new QMenu();
  menu_right_mouse_full->addAction(action_add_roi);
  menu_right_mouse_full->addAction(action_load_img);
  menu_right_mouse_full->addAction(action_remove_img);
  menu_right_mouse_full->addAction(action_reset_img);

  menu_right_mouse_roi = new QMenu();
  menu_right_mouse_roi->addAction(action_save_roi);
  menu_right_mouse_roi->addAction(action_delete_roi);
}

bool ImgRoiWidget::hadImage() {
  return (m_pixmapItem != nullptr);
}

QString ImgRoiWidget::interactMode2String(InteractMode mode) {
  switch (mode) {
    case IModeNone:
      return "None";
    case IModeZoom:
      return "Zoom";
    case IModePan:
      return "Pan";
    case IModeDrawROI:
      return "Draw ROI";
  }
  return "Unknown";
}

bool ImgRoiWidget::rightMouseButtonPressed(QMouseEvent *event) {
  return false;
}

bool ImgRoiWidget::leftMouseButtonPressed(QMouseEvent *event) {
  if ((event->modifiers() & Qt::ControlModifier) && (!m_roi_started)) {
    m_has_panned = false;
    m_last_pan_point = event->pos();
    changeInteractMode(IModePan);
    return false;
  }

  if (m_current_mode == IModeDrawROI) {
    if (event->modifiers() != Qt::NoModifier) {
      return false;
    }

    qDebug() << "[IMG ROI Widget] Add new ROI: first point marked.";

    m_roi_start_point = mapToScene(event->pos());
    m_temp_roi = new QGraphicsRectItem();
    m_temp_roi->setPen(QPen(Qt::red, 2, Qt::DashLine));
    m_temp_roi->setBrush(QBrush(Qt::transparent));
    m_temp_roi->setRect(QRectF(m_roi_start_point, QSizeF(0, 0)));

    if (this->scene()) {
      this->scene()->addItem(m_temp_roi);
    }
    return true;
  }

  return false;
}

bool ImgRoiWidget::rightMouseButtonReleased(QMouseEvent *event) {
  showRightMouseClickMenu(event);
  return false;
}

bool ImgRoiWidget::leftMouseButtonReleased(QMouseEvent *event) {
  switch (m_current_mode) {
    case IModeNone:
      {

      }
      break;
    case IModeZoom:

      break;
    case IModePan:
      {
        m_last_pan_point = QPoint();
        backToPreviousMode();
        if (m_has_panned) {
          return true;
        } else {
          return false;
        }
      }
      break;
    case IModeDrawROI:
      {
        if (this->mapToScene(event->pos()) == m_roi_start_point) {
          this->scene()->removeItem(m_temp_roi);
          m_temp_roi = nullptr;
          return true;
        }

        if (this->scene()) {
          temp_editable_roi = new RoiItem(m_temp_roi->rect(), this->m_pixmapItem, &m_scene_interacting);
          QPointF temp_pos = m_temp_roi->pos();
          this->scene()->removeItem(m_temp_roi);
          m_temp_roi = nullptr;

          if ((temp_editable_roi->rect().width() < 10) || (temp_editable_roi->rect().height() < 10)) {
            qDebug() << "[IMG ROI Widget] Add new ROI: failed, ROI to small.";
          } else {
            temp_editable_roi->setPos(temp_pos);
            this->scene()->addItem(temp_editable_roi);
            qDebug() << "[IMG ROI Widget] Add new ROI: finished.";
          }
        }
        changeInteractMode(IModeNone);
        return true;
      }
      break;
    default:
      break;
  }

  return false;
}

void ImgRoiWidget::showRightMouseClickMenu(QMouseEvent *event) {
  QMenu *showMenu = nullptr;

  if (!m_scene->selectedItems().empty()) {
    showMenu = menu_right_mouse_roi;
  } else {
    if (hadImage()) {
      showMenu = menu_right_mouse_full;
    } else {
      showMenu = menu_right_mouse;
    }
  }

  QAction *selectedAction = showMenu->exec(event->globalPosition().toPoint());

  if (selectedAction == action_add_roi) {
    qDebug() << "[IMG ROI Widget] User choose add new ROI from mouse event.";
    startDrawROI();

  } else if (selectedAction == action_delete_roi) {
    deletedSelectedItems();
    qDebug() << "[IMG ROI Widget] User choose delete selected ROIs from mouse event.";

  } else if (selectedAction == action_save_roi) {
    // deletedSelectedItems();
    qDebug() << "[IMG ROI Widget] User choose save selected ROIs from mouse event.";

    QPointF scene_pos = mapToScene(event->pos());
    QGraphicsItem *item = this->scene()->itemAt(scene_pos, QTransform());
    // qDebug() << item->type();

    if (item) {
      RoiItem *roi_item = (RoiItem*)item;
      if (m_scene->selectedItems().contains(item)) {
        // qDebug() << "User clicked at item:"
        //          << roi_item->rect().topLeft()
        //          << roi_item->rect().bottomRight();

        const QPixmap &pixmap = m_pixmapItem->pixmap();

        QRectF save_rect = roi_item->rect();
        QRectF roiInPixmapItem = m_pixmapItem->mapFromScene(save_rect).boundingRect();
        QRect roi = save_rect.toAlignedRect();
        roi = roi.intersected(pixmap.rect());
        if (roi.isEmpty())
          return;

        QPixmap cropped = pixmap.copy(roi);

        if (m_setting == nullptr) {
          m_setting = new QSettings("DGB", "Image widget");
        }

        QString lastDirectory = m_setting->value("lastDirectory", "").toString();

        QString filePath = QFileDialog::getSaveFileName(this,
                                                        "Save image",
                                                        lastDirectory,
                                                        "Image Files (*.png *.jpg *.jpeg *.bmp)");

        if (!filePath.isEmpty()) {
          QString directory = QFileInfo(filePath).absolutePath();
          m_setting->setValue("lastDirectory", directory);

          cropped.save(filePath, "bmp");
        }


      }
    }

  } else if (selectedAction == action_load_img) {
    qDebug() << "[IMG ROI Widget] User choose load image from mouse event.";
    showChooseImageDialog();

  } else if (selectedAction == action_remove_img) {
    qDebug() << "[IMG ROI Widget] User choose remove image from mouse event.";
    this->removeImage();

  } else if (selectedAction == action_reset_img) {
    qDebug() << "[IMG ROI Widget] User choose reset image transform from mouse event.";

    if (!hadImage()) {
      qDebug() << "[IMG ROI Widget] Rest transform fail, image empty.";
      return;
    }
    this->fitInView(m_pixmap_bounding_rect, Qt::KeepAspectRatio);
  }
}

void ImgRoiWidget::showChooseImageDialog() {
  if (m_setting == nullptr) {
    m_setting = new QSettings("DGB", "Image widget");
  }

  QString lastDirectory = m_setting->value("lastDirectory", "").toString();

  QString filePath = QFileDialog::getOpenFileName(this,
                                                  "Choose image",
                                                  lastDirectory,
                                                  "Image Files (*.png *.jpg *.jpeg *.bmp)");

  if (!filePath.isEmpty()) {
    QString directory = QFileInfo(filePath).absolutePath();
    m_setting->setValue("lastDirectory", directory);

    loadImage(filePath);
  }
}
