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
import matplotlib.cm as cm


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

        # get representative image
        self.pixmap = self.parent.spot_ui.circles.get_representative_image()

        # Create a QGraphicsPixmapItem and set the pixmap
        self.pixmap_item = QGraphicsPixmapItem(self.pixmap)
        self.graphics_scene.addItem(self.pixmap_item)

        # Set the size policy and scaling mode
        self.graphics_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.graphics_view.fitInView(self.pixmap_item, Qt.KeepAspectRatio)

        # Add the CustomGraphicsView to the layout
        self.layout.addWidget(self.graphics_view)

    def setData(self, spots, binding_probability):
        self.spots = spots
        self.binding_probability = binding_probability

    def draw(self):

        cmap = cm.get_cmap("viridis")

        # Create a QPainter to draw on the pixmap
        painter = QPainter(self.pixmap)
        pen = QPen(Qt.NoPen)  # No outline for the circles
        painter.setPen(pen)

        # Draw red filled circles at the spot positions
        for spot, prob in zip(self.spots, self.binding_probability):

            x, y, r = spot
            rgba = cmap(prob)
            color = QColor(
                int(rgba[0] * 255),
                int(rgba[1] * 255),
                int(rgba[2] * 255),
                int(rgba[3] * 255),
            )
            brush = QBrush(color)
            painter.setBrush(brush)
            painter.drawEllipse(x - r, y - r, 2 * r, 2 * r)

        painter.end()  # Finish painting

        # Update the pixmap in the QGraphicsPixmapItem
        self.pixmap_item.setPixmap(self.pixmap)

        # Update the scene with the new pixmap
        self.graphics_scene.update()
