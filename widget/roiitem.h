#ifndef ROIITEM_H
#define ROIITEM_H

#include <QGraphicsRectItem>
#include <QPen>
#include <QBrush>
#include <QCursor>
#include <QPainter>
#include <QStyleOptionGraphicsItem>
#include <QGraphicsSceneMouseEvent>
#include <QtMath>

class RoiItem : public QGraphicsRectItem {
  public:
    enum { Type = UserType + 27 };
    enum HandlePosition {
      None,
      TopLeft,
      TopRight,
      BottomLeft,
      BottomRight,
      RotateHandle };

    RoiItem(const QRectF &rect, QGraphicsItem *parent = nullptr, bool *ignore_flag = nullptr);
    int type() const override { return Type; }

  protected:
    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget) override;
    QRectF boundingRect() const override;
    QPainterPath shape() const override;
    QVariant itemChange(QGraphicsItem::GraphicsItemChange change, const QVariant &value) override;
    QRectF handleRect(HandlePosition pos) const;
    QRectF rotateHandleRect() const;
    HandlePosition getHandleAt(const QPointF &pos) const;

    void mousePressEvent(QGraphicsSceneMouseEvent *event) override;
    void mouseMoveEvent(QGraphicsSceneMouseEvent *event) override;
    void mouseReleaseEvent(QGraphicsSceneMouseEvent *event) override;

  private:
    // handle point size
    qreal m_handle_size;
    // offset to draw handle
    qreal m_rotation_handle_offset;
    HandlePosition m_current_handle;
    // variables for resize
    QRectF m_original_rect;
    QPointF m_press_pos;
    // variables for rotate
    qreal m_original_rotation;
    QPointF m_rotation_origin;
    qreal m_press_angle;

    QPointF m_last_center;

    bool *m_ignore;
};
#endif // ROIITEM_H
