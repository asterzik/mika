from PySide6.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QGraphicsScene,
    QGraphicsPixmapItem,
    QSizePolicy,
    QVBoxLayout,
    QLabel,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QLinearGradient, QFont
from misc.graphics_view import CustomGraphicsView
import matplotlib.cm as cm
import numpy as np

cmap = cm.get_cmap("viridis")


class ColorMapLegend(QWidget):
    def __init__(self, width=80, height=600, parent=None):
        super().__init__(parent)
        self.setFixedSize(width, height)
        self.setFixedSize(width, height)
        self.setMouseTracking(True)
        self.setToolTip(
            """
                <b>Binding Score:</b> <br>
                |mean(range1) - mean(range2)| / (3 × std) <br><br>
                <i>Used to estimate strength of signal difference.</i>
            """
        )
        self.cmap = cm.get_cmap("viridis")

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w, h = self.width(), self.height()

        # --- Sizes ---
        text_margin = 4
        label_height = 14
        gradient_top = label_height + text_margin
        gradient_bottom = h - label_height - text_margin
        gradient_height = gradient_bottom - gradient_top

        painter.setPen(Qt.black)
        painter.drawText(0, 0, w, label_height, Qt.AlignHCenter | Qt.AlignBottom, "1")

        # --- Draw gradient ---
        gradient = QLinearGradient(0, gradient_top, 0, gradient_bottom)
        for i in range(100):
            pos = i / 99
            r, g, b, a = self.cmap(pos)
            gradient.setColorAt(
                1.0 - pos,
                QColor(int(r * 255), int(g * 255), int(b * 255), int(a * 255)),
            )

        painter.fillRect(5, gradient_top, w - 10, gradient_height, gradient)

        # --- Draw bottom label ---
        painter.drawText(
            0,
            h - label_height,
            w,
            label_height,
            Qt.AlignHCenter | Qt.AlignTop,
            "0",
        )


class ResultsView:
    def __init__(self, parent=None):
        self.parent = parent

        # Create the main widget and layout
        self.widget = QWidget()
        self.main_layout = (
            QVBoxLayout()
        )  # Use a vertical layout for the title and content
        self.widget.setLayout(self.main_layout)

        # Add the title at the top
        title_label = QLabel("Binding Score")
        title_label.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(title_label)

        # Create the content layout (image and colormap legend)
        content_layout = QHBoxLayout()

        # Add the graphics view (image display)
        self.graphics_view = CustomGraphicsView(self)
        self.graphics_scene = QGraphicsScene(self.graphics_view)
        self.graphics_view.setScene(self.graphics_scene)

        # Get representative image
        self.pixmap = self.parent.spot_ui.circles.get_representative_image()

        # Create a QGraphicsPixmapItem and set the pixmap
        self.pixmap_item = QGraphicsPixmapItem(self.pixmap)
        self.graphics_scene.addItem(self.pixmap_item)

        # Set the size policy and scaling mode
        self.graphics_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.graphics_view.fitInView(self.pixmap_item, Qt.KeepAspectRatio)

        # Add the graphics view to the content layout
        content_layout.addWidget(self.graphics_view)

        # Add the colormap legend to the content layout
        legend = ColorMapLegend()
        content_layout.addWidget(legend)

        # Add the content layout to the main layout
        self.main_layout.addLayout(content_layout)

    def setData(self, spots, binding_probability):
        self.spots = spots
        self.binding_probability = binding_probability

    def draw(self):
        # Create a QPainter to draw on the pixmap
        painted_pixmap = self.pixmap.copy()
        painter = QPainter(painted_pixmap)
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
        self.pixmap_item.setPixmap(painted_pixmap)

        # Update the scene with the new pixmap
        self.graphics_scene.update()
