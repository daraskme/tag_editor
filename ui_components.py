# pyre-ignore-all-errors[21]
from PyQt6.QtWidgets import QLayout, QLayoutItem, QSizePolicy, QPushButton, QLabel, QStyle, QWidget
from PyQt6.QtCore import Qt, QSize, QPoint, QRect, pyqtSignal

class FlowLayout(QLayout):
    def __init__(self, parent=None, margin=-1, hSpacing=-1, vSpacing=-1):
        super().__init__(parent) # type: ignore
        self.m_hSpace = hSpacing
        self.m_vSpace = vSpacing
        self.itemList: list[QLayoutItem] = []
        self.setContentsMargins(margin, margin, margin, margin)

    def __del__(self):
        item = self.takeAt(0)
        while item:
            item = self.takeAt(0)

    def addItem(self, item):
        self.itemList.append(item)

    def horizontalSpacing(self):
        if self.m_hSpace >= 0:
            return self.m_hSpace
        else:
            return self.smartSpacing(QStyle.PixelMetric.PM_LayoutHorizontalSpacing)

    def verticalSpacing(self):
        if self.m_vSpace >= 0:
            return self.m_vSpace
        else:
            return self.smartSpacing(QStyle.PixelMetric.PM_LayoutVerticalSpacing)

    def count(self):
        return len(self.itemList)

    def itemAt(self, index):
        if 0 <= index < len(self.itemList):
            return self.itemList[index]
        return None

    def takeAt(self, index):
        if 0 <= index < len(self.itemList):
            return self.itemList.pop(index)
        return None

    def expandingDirections(self):
        return Qt.Orientation(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        height = self.doLayout(QRect(0, 0, width, 0), True)
        return height

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self.doLayout(rect, False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QSize()
        for item in self.itemList:
            size = size.expandedTo(item.minimumSize())
        margin, _, _, _ = self.getContentsMargins()
        # pyre-ignore[16, 58]
        size += QSize(2 * margin, 2 * margin) # type: ignore
        return size

    def doLayout(self, rect, testOnly):
        x = rect.x()
        y = rect.y()
        lineHeight = 0

        for item in self.itemList:
            wid = item.widget()
            spaceX = self.horizontalSpacing()
            if spaceX == -1:
                spaceX = wid.style().layoutSpacing(QSizePolicy.ControlType.PushButton, QSizePolicy.ControlType.PushButton, Qt.Orientation.Horizontal) if wid else 0
                
            spaceY = self.verticalSpacing()
            if spaceY == -1:
                spaceY = wid.style().layoutSpacing(QSizePolicy.ControlType.PushButton, QSizePolicy.ControlType.PushButton, Qt.Orientation.Vertical) if wid else 0

            nextX = x + item.sizeHint().width() + spaceX
            if nextX - spaceX > rect.right() and lineHeight > 0:
                x = rect.x()
                y = y + lineHeight + spaceY
                nextX = x + item.sizeHint().width() + spaceX
                lineHeight = 0

            if not testOnly:
                item.setGeometry(QRect(QPoint(x, y), item.sizeHint()))

            x = nextX
            lineHeight = max(lineHeight, item.sizeHint().height())

        return y + lineHeight - rect.y()

    def smartSpacing(self, pm):
        parent = self.parent()
        if not parent:
            return -1
        elif parent.isWidgetType():
            return parent.style().pixelMetric(pm, None, parent)
        else:
            return parent.spacing()

class TagButton(QPushButton):
    deleted = pyqtSignal(str)
    edit_requested = pyqtSignal(str)

    def __init__(self, tag_text, parent=None):
        super().__init__(tag_text, parent) # type: ignore
        self.tag_text = tag_text
        self.setStyleSheet("QPushButton { border-radius: 10px; background-color: #3498db; color: white; padding: 5px 10px; font-weight: bold; } QPushButton:hover { background-color: #e74c3c; }")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Left click to remove, Right click to edit")

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.deleted.emit(self.tag_text)
            self.deleteLater()
        elif event.button() == Qt.MouseButton.RightButton:
            self.edit_requested.emit(self.tag_text)
        super().mousePressEvent(event)

class ClickableImageLabel(QLabel):
    clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent) # type: ignore
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setScaledContents(False) # We will scale pixmap manually

    def mousePressEvent(self, event):
        self.clicked.emit()
        super().mousePressEvent(event)

class FlowContainer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent) # type: ignore
        self._layout = None

    def setLayout(self, layout):
        super().setLayout(layout)
        self._layout = layout

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._layout:
            # Force the layout to recalculate its height based on the new width
            # pyre-ignore-all-errors[16]
            height = self._layout.heightForWidth(event.size().width())
            self.setMinimumHeight(height)
