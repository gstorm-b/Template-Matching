#include "roiitem.h"

RoiItem::RoiItem(const QRectF &rect, QGraphicsItem *parent,
                 bool *ignore_flag)
    : QGraphicsRectItem(rect, parent),
      m_current_handle(None) {

  setFlags(QGraphicsItem::ItemIsMovable |
           QGraphicsItem::ItemIsSelectable |
           QGraphicsItem::ItemSendsGeometryChanges);

  m_ignore = ignore_flag;

  m_handle_size = 20.0;
  m_rotation_handle_offset = 2.0;
  this->setTransformOriginPoint(rect.center());
}

void RoiItem::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget) {
  Q_UNUSED(widget);
  // Draw ROI
  QPen pen(Qt::red);
  pen.setWidth(2);
  painter->setPen(pen);
  painter->setBrush(Qt::NoBrush);
  painter->drawEllipse(this->rect().center(), 5, 5);
  painter->drawEllipse(this->transformOriginPoint(), 10, 10);
  // qDebug() << rect().center() << this->transformOriginPoint();

  // pen.setColor(0xC68EFD);
  // pen.setStyle(Qt::DashLine);
  // painter->setPen(pen);
  // painter->drawRect(rect());

  // if ROI is selected draw Handle point
  if (option->state & QStyle::State_Selected) {
    pen.setColor(0xEC5228);
    pen.setStyle(Qt::DashLine);
    painter->setPen(pen);
    painter->drawRect(rect());

    QColor handler_color(0x8F87F1);
    handler_color.setAlpha(100);
    painter->setBrush(handler_color);
    painter->setPen(Qt::black);
    // draw handle point add 4 corner
    painter->drawRect(handleRect(TopLeft));
    painter->drawRect(handleRect(TopRight));
    painter->drawRect(handleRect(BottomLeft));
    painter->drawRect(handleRect(BottomRight));

    // Vẽ handle xoay (vẽ dưới dạng hình ellipse, màu xanh)
    // painter->setBrush(Qt::green);
    painter->drawEllipse(rotateHandleRect());
    // Tùy chọn: vẽ một đường nối từ cạnh trên của ROI đến handle xoay
    QPointF topCenter = QPointF(rect().center().x(), rect().top());
    painter->setPen(QPen(Qt::blue, 1, Qt::DashLine));
    painter->drawLine(topCenter, rotateHandleRect().center());
  } else {
    pen.setColor(0xC68EFD);
    pen.setStyle(Qt::DashLine);
    painter->setPen(pen);
    painter->drawRect(rect());
  }
}

// Trong lớp EditableROIItem
QRectF RoiItem::boundingRect() const {
  QRectF base = rect();
  // Mở rộng boundingRect bằng cách hợp nhất (union)
  QRectF unionRect = base;
  unionRect = unionRect.united(handleRect(TopLeft));
  unionRect = unionRect.united(handleRect(TopRight));
  unionRect = unionRect.united(handleRect(BottomLeft));
  unionRect = unionRect.united(handleRect(BottomRight));
  unionRect = unionRect.united(rotateHandleRect());
  // Có thể thêm thêm một chút margin nếu cần (ví dụ: 1-2 pixels)
  return unionRect.adjusted(-1, -1, 1, 1);
}

QPainterPath RoiItem::shape() const  {
  QPainterPath path;
  path.setFillRule(Qt::WindingFill);
  // Bao gồm phần chính của item (rect)
  // path.addRect(rect().adjusted(m_handle_size/2, m_handle_size/2, -m_handle_size/2, -m_handle_size/2));
  path.addRect(rect());

  // Nếu item được chọn, thêm các handle vào vùng shape để tăng vùng nhận sự kiện
  if (isSelected()) {
    path.addRect(handleRect(TopLeft));
    path.addRect(handleRect(TopRight));
    path.addRect(handleRect(BottomLeft));
    path.addRect(handleRect(BottomRight));
    // Nếu có handle xoay, cũng có thể thêm handle xoay vào đó:
    path.addEllipse(rotateHandleRect());
  }
  return path;
}

QVariant RoiItem::itemChange(QGraphicsItem::GraphicsItemChange change, const QVariant &value) {
  // if (change == QGraphicsItem::) {
  //   // Lấy rect mới và cập nhật transform origin
  //   QRectF newRect = value.toRectF();
  //   setTransformOriginPoint(newRect.center());
  // }

  if (change == QGraphicsItem::ItemPositionChange && parentItem()) {
    QPointF newPos = value.toPointF();

    // bounding rect của chính item trong hệ toạ độ của parent
    // QRectF rectInParent = mapRectToParent(boundingRect());
    QRectF rectInParent = mapRectToParent(rect());
    QRectF parentRect = parentItem()->boundingRect();

    // vector offset khi di chuyển
    QPointF delta = newPos - pos();
    QRectF movedRect = rectInParent.translated(delta);

    // nếu vượt ra khỏi parent thì điều chỉnh lại
    QPointF corrected = newPos;

    // if (correctRectItem(movedRect, corrected)) {
    //   return corrected;
    // }

    if (!parentRect.contains(movedRect)) {
      if (movedRect.left() < parentRect.left())
        corrected.rx() += parentRect.left() - movedRect.left();
      if (movedRect.right() > parentRect.right())
        corrected.rx() -= movedRect.right() - parentRect.right();
      if (movedRect.top() < parentRect.top())
        corrected.ry() += parentRect.top() - movedRect.top();
      if (movedRect.bottom() > parentRect.bottom())
        corrected.ry() -= movedRect.bottom() - parentRect.bottom();

      return corrected;
    }
  }

  return QGraphicsRectItem::itemChange(change, value);
}

QRectF RoiItem::handleRect(HandlePosition pos) const {
  QRectF r = rect();
  QPointF point;
  switch(pos) {
    case TopLeft:
      point = r.topLeft();
      break;
    case TopRight:
      point = r.topRight();
      break;
    case BottomLeft:
      point = r.bottomLeft();
      break;
    case BottomRight:
      point = r.bottomRight();
      break;
    default:
      point = QPointF();
      break;
  }
  // draw rectangle handle with center at corner
  return QRectF(point.x() - m_handle_size/2, point.y() - m_handle_size/2,
                m_handle_size, m_handle_size);
}

QRectF RoiItem::rotateHandleRect() const {
  QRectF r = rect();
  // Tính điểm giữa của cạnh trên
  QPointF topCenter(r.center().x(), r.top());
  // Đẩy ra phía trên với độ offset định trước
  QPointF handleCenter = topCenter - QPointF(0, m_rotation_handle_offset);
  return QRectF(handleCenter.x() - m_handle_size/2.0, handleCenter.y() - m_handle_size/2.0,
                m_handle_size, m_handle_size);
}

RoiItem::HandlePosition RoiItem::getHandleAt(const QPointF &pos) const {
  if (handleRect(TopLeft).contains(pos))
    return TopLeft;
  if (handleRect(TopRight).contains(pos))
    return TopRight;
  if (handleRect(BottomLeft).contains(pos))
    return BottomLeft;
  if (handleRect(BottomRight).contains(pos))
    return BottomRight;
  if (rotateHandleRect().contains(pos))
    return RotateHandle;
  return None;
}

bool RoiItem::isInsideParent(QRectF &new_rect) {
  if (parentItem() != nullptr) {
    QRectF childInParent = this->mapRectToParent(new_rect);
    QRectF parentRect = parentItem()->boundingRect();
    return parentRect.contains(childInParent);
  }
  return false;
}

void RoiItem::mousePressEvent(QGraphicsSceneMouseEvent *event) {
  if (m_ignore != nullptr) {
    if (*m_ignore == true) {
      event->ignore();
      return;
    }
  }

  // if(!isSelected()) {
  //   event->ignore();
  //   QGraphicsRectItem::mousePressEvent(event);
  // }
  // qDebug() << "[ROI Item] Into resize state";

  QPointF pos = event->pos();
  m_last_center = this->transformOriginPoint();
  m_current_handle = getHandleAt(pos);
  if (m_current_handle == RotateHandle) {
    // Lưu lại góc xoay ban đầu và tâm xoay (được thiết lập từ transformOriginPoint)
    m_original_rotation = rotation();
    m_valid_rotation = m_original_rotation;
    m_rotation_origin = mapToScene(transformOriginPoint());
    // m_rotation_origin = mapToScene(rect().center());
    // Tính góc ban đầu từ tâm đến vị trí chuột
    m_press_angle = QLineF(m_rotation_origin, event->scenePos()).angle();
    setTransformOriginPoint(rect().center());
    event->accept();
    return;
  } else if (m_current_handle != None) {
    // Nếu nhấn vào handle, lưu lại rect ban đầu
    m_original_rect = rect();
    m_press_pos = pos;
    event->accept();
    return;
  }

  // Nếu nhấn vào bên trong, chuyển sang chế độ di chuyển (đã được xử lý bởi QGraphicsItem::ItemIsMovable)
  event->accept();
  QGraphicsRectItem::mousePressEvent(event);
}

void RoiItem::mouseMoveEvent(QGraphicsSceneMouseEvent *event) {
  if (m_ignore != nullptr) {
    if (*m_ignore == true) {
      event->ignore();
      return;
    }
  }

  if (m_current_handle == RotateHandle) {
    // Tính góc hiện tại giữa tâm và vị trí chuột
    qreal currentAngle = QLineF(m_rotation_origin, event->scenePos()).angle();
    qreal angleDiff = currentAngle - m_press_angle;
    setRotation(m_original_rotation - angleDiff);

    // avoid rotate out side of parrent
    QRectF new_rect = mapRectToParent(rect());
    if (!parentItem()->boundingRect().contains(new_rect)) {
      setRotation(m_valid_rotation);
    } else {
      m_valid_rotation = m_original_rotation - angleDiff;
    }

    event->accept();
    return;
  } else if (m_current_handle != None) {
    // Tính delta di chuyển
    QPointF delta = event->pos() - m_press_pos;
    QRectF newRect = m_original_rect;
    switch(m_current_handle) {
      case TopLeft:
        newRect.setTopLeft(newRect.topLeft() + delta);
        break;
      case TopRight:
        newRect.setTopRight(newRect.topRight() + delta);
        break;
      case BottomLeft:
        newRect.setBottomLeft(newRect.bottomLeft() + delta);
        break;
      case BottomRight:
        newRect.setBottomRight(newRect.bottomRight() + delta);
        break;
      default:
        break;
    }

    // check new position outside parrent
    if (!isInsideParent(newRect)) {
      event->accept();
      return;
    }


    // Đảm bảo rằng newRect có kích thước hợp lý
    if (newRect.width() < m_handle_size*2) {
      // newRect.setWidth(m_handle_size*2);
      // setTransformOriginPoint(rect().center());
      event->accept();
      return;
    }

    if (newRect.height() < m_handle_size*2) {
      // newRect.setHeight(m_handle_size*2);
      // setTransformOriginPoint(rect().center());
      event->accept();
      return;
    }

    setRect(newRect);
    // setTransformOriginPoint(rect().center());
    event->accept();
    return;
  }

  event->accept();
  QGraphicsRectItem::mouseMoveEvent(event);
}

void RoiItem::mouseReleaseEvent(QGraphicsSceneMouseEvent *event) {
  if (m_ignore != nullptr) {
    if (*m_ignore == true) {
      event->ignore();
      return;
    }
  }

  if (m_current_handle != RotateHandle) {
    QPointF oldCenterScene = this->mapToScene(this->rect().center());
    this->setTransformOriginPoint(this->rect().center());
    QPointF newCenterScene = this->mapToScene(this->rect().center());
    this->setPos(this->pos() + oldCenterScene - newCenterScene);
  }
  m_current_handle = None;
  event->accept();
  QGraphicsRectItem::mouseReleaseEvent(event);
}

