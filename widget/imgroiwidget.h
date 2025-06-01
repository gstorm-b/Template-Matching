#ifndef IMGROIWIDGET_H
#define IMGROIWIDGET_H

#include <QMouseEvent>
#include <QWheelEvent>
#include <QWidget>
#include <QGraphicsView>
#include <QGraphicsScene>
#include <QSettings>
#include <QPointF>
#include <QGraphicsItemGroup>
#include <QGraphicsPixmapItem>
#include <QAction>

#include "widget/roiitem.h"

class ImgRoiWidget : public QGraphicsView {
    Q_OBJECT

  public:
    enum InteractMode {
      IModeNone,
      IModeZoom,
      IModePan,
      IModeDrawROI
    };

    explicit ImgRoiWidget(QWidget *parent = nullptr);
    void setSettings(QSettings *setting);
    void removeSettings();
    QPixmap getImage();

  public slots:
    void loadImage(const QString &filePath);
    void loadImage(QPixmap &pixmap);
    void removeImage();
    void startDrawROI();
    void deletedSelectedItems();

  protected:
    void mousePressEvent(QMouseEvent *event) override;
    void mouseMoveEvent(QMouseEvent *event) override;
    void mouseReleaseEvent(QMouseEvent *event) override;
    void mouseDoubleClickEvent(QMouseEvent *event) override;
    void wheelEvent(QWheelEvent *event) override;
    void keyPressEvent(QKeyEvent *event) override;
    void keyReleaseEvent(QKeyEvent *event) override;

  private:
    void init_mouse_menu();

    bool hadImage();

    void changeInteractMode(InteractMode mode);
    void backToPreviousMode();
    void changeCursor();
    QString interactMode2String(InteractMode mode);

    bool rightMouseButtonPressed(QMouseEvent *event);
    bool leftMouseButtonPressed(QMouseEvent *event);
    bool rightMouseButtonReleased(QMouseEvent *event);
    bool leftMouseButtonReleased(QMouseEvent *event);

    void showRightMouseClickMenu(QMouseEvent *event);
    void showChooseImageDialog();

  private:
    QGraphicsScene *m_scene;
    QGraphicsPixmapItem *m_pixmapItem;
    QRectF m_pixmap_bounding_rect;

    QSettings *m_setting;
    bool m_using_user_setting;

    InteractMode m_current_mode;
    InteractMode m_previous_mode;
    bool m_scene_interacting;

    bool m_has_panned;
    QPoint m_last_pan_point;

    bool m_roi_started;
    QGraphicsRectItem *m_temp_roi;
    QPointF m_roi_start_point;
    RoiItem *temp_editable_roi;

    /// Right Click Menu
    QMenu *menu_right_mouse;
    QMenu *menu_right_mouse_full;
    QMenu *menu_right_mouse_roi;
    QAction *action_add_roi;
    QAction *action_delete_roi;
    QAction *action_save_roi;
    QAction *action_load_img;
    QAction *action_remove_img;
    QAction *action_reset_img;
};

#endif // IMGROIWIDGET_H
