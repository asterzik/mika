from PySide6.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QGraphicsScene,
    QGraphicsPixmapItem,
    QSizePolicy,
    QVBoxLayout,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QPen, QColor, QBrush
from misc.graphics_view import CustomGraphicsView


class ResultsView:
    def __init__(self, parent=None):
        self.parent = parent
        self.widget = QWidget()
        self.layout = QHBoxLayout()
        self.widget.setLayout(self.layout)

        self.graphics_view = CustomGraphicsView(self)
        self.graphics_view = CustomGraphicsView(self)
        self.graphics_scene = QGraphicsScene(self.graphics_view)
        self.graphics_view.setScene(self.graphics_scene)

        # Extract the pixmap from the interactive image
        self.pixmap = self.parent.spot_ui.interactive_image.pixmap_item.pixmap()

        # Create a QGraphicsPixmapItem and set the pixmap
        self.pixmap_item = QGraphicsPixmapItem(self.pixmap)
        self.graphics_scene.addItem(self.pixmap_item)

        # Set the size policy and scaling mode
        self.graphics_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.graphics_view.fitInView(self.pixmap_item, Qt.KeepAspectRatio)

        # Add the CustomGraphicsView to the layout
        self.layout.addWidget(self.graphics_view)

    def setData(self, spots, labels, label_list, diff, std):
        self.spots = spots
        self.labels = labels
        self.label_list = label_list
        self.diff = diff
        self.std = std

    def draw(self):
        # Ensure the pixmap is up-to-date
        self.pixmap = self.parent.spot_ui.interactive_image.pixmap_item.pixmap()

        # Create a QPainter to draw on the pixmap
        painter = QPainter(self.pixmap)
        pen = QPen(Qt.NoPen)  # No outline for the circles
        painter.setPen(pen)
        brush = QBrush(QColor(255, 0, 0, 127))  # Semi-transparent red
        painter.setBrush(brush)

        # Draw red filled circles at the spot positions
        for spot, labels in zip(self.spots, self.labels):

            x, y, r = spot
            painter.drawEllipse(x - r, y - r, 2 * r, 2 * r)

        painter.end()  # Finish painting

        # Update the pixmap in the QGraphicsPixmapItem
        self.pixmap_item.setPixmap(self.pixmap)

        # Update the scene with the new pixmap
        self.graphics_scene.update()
