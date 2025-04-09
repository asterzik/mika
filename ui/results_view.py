from PySide6.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QGraphicsScene,
    QGraphicsPixmapItem,
    QSizePolicy,
    QVBoxLayout,
    QLabel,
    QLineEdit,
    QGroupBox,
    QRadioButton,
    QSpacerItem,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QLinearGradient, QFont
from misc.graphics_view import CustomGraphicsView
import matplotlib.cm as cm
import numpy as np

cmap = cm.get_cmap("viridis")


class ColorMapLegend(QWidget):
    def __init__(self, color_image, width=80, height=600, parent=None):
        self.color_image = color_image
        super().__init__(parent)
        self.setFixedSize(width, height)

        self.cmap = cm.get_cmap("viridis")
        self.lower_bound = 0
        self.upper_bound = 1
        self.crit = None

        # Create input fields
        self.upper_input = QLineEdit(str(self.upper_bound), self)
        self.lower_input = QLineEdit(str(self.lower_bound), self)

        self.upper_input.setFixedHeight(20)
        self.lower_input.setFixedHeight(20)
        self.upper_input.setAlignment(Qt.AlignCenter)
        self.lower_input.setAlignment(Qt.AlignCenter)

        # Positioning
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self.upper_input)
        layout.addStretch(1)
        layout.addWidget(self.lower_input)

        # Connect signals
        self.upper_input.editingFinished.connect(self.boundsChanged)
        self.lower_input.editingFinished.connect(self.boundsChanged)

    def boundsChanged(self):
        upper = float(self.upper_input.text())
        lower = float(self.lower_input.text())
        self.upper_bound = upper
        self.lower_bound = lower
        self.color_image.updateBounds(lower, upper)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w, h = self.width(), self.height()
        top = self.upper_input.height() + 4
        bottom = h - self.lower_input.height() - 4
        gradient_height = bottom - top

        # Draw gradient
        gradient = QLinearGradient(0, top, 0, bottom)
        for i in range(100):
            pos = i / 99
            r, g, b, a = self.cmap(pos)
            gradient.setColorAt(
                1.0 - pos,
                QColor(int(r * 255), int(g * 255), int(b * 255), int(a * 255)),
            )

        painter.fillRect(5, top, w - 10, gradient_height, gradient)

        # Draw critical point if it's within bounds
        if self.crit is not None and self.lower_bound < self.crit < self.upper_bound:
            rel_pos = (self.crit - self.lower_bound) / (
                self.upper_bound - self.lower_bound
            )
            y_pos = top + (1.0 - rel_pos) * gradient_height
            pen = QPen(Qt.black)
            pen.setWidth(2)
            painter.setPen(pen)
            painter.drawLine(0, int(y_pos), w, int(y_pos))

    def updateBounds(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.upper_input.setText(str(self.upper_bound))
        self.lower_input.setText(str(self.lower_bound))

    def updateCriticalPoint(self, crit):
        self.crit = crit
        self.update()


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
        self.content_layout = QHBoxLayout()

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
        self.content_layout.addWidget(self.graphics_view)

        # Add the colormap legend to the content layout
        self.legend = ColorMapLegend(self)
        self.content_layout.addWidget(self.legend)

        # Add the content layout to the main layout
        self.main_layout.addLayout(self.content_layout)

        # --- Radio Buttons for Color Mode ---
        self.color_mode_group = QGroupBox("Coloring Mode")
        self.color_mode_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

        self.abs_radio = QRadioButton("Abs. Difference")
        self.std_radio = QRadioButton("Std. Deviation")
        self.binary_radio = QRadioButton("Binary Class.")
        self.abs_radio.setChecked(True)

        radio_layout = QVBoxLayout()
        radio_layout.setContentsMargins(6, 6, 6, 6)
        radio_layout.setSpacing(4)  # Compact spacing between options
        radio_layout.addWidget(self.abs_radio)
        radio_layout.addWidget(self.std_radio)
        radio_layout.addWidget(self.binary_radio)

        self.color_mode_group.setLayout(radio_layout)
        self.content_layout.addWidget(self.color_mode_group)

        self.abs_radio.toggled.connect(self.onColorModeChanged)
        self.std_radio.toggled.connect(self.onColorModeChanged)
        self.binary_radio.toggled.connect(self.onColorModeChanged)

    def onColorModeChanged(self):
        if self.abs_radio.isChecked():
            self.value = self.diff
            self.lower = round(self.min, 2)
            self.upper = round(self.max, 2)
            self.legend.updateBounds(self.lower, self.upper)
            self.legend.updateCriticalPoint(3 * np.mean(self.diff_std))
            self.draw()
        elif self.std_radio.isChecked():
            self.value = self.diff_std
            self.lower = round(np.min(self.diff_std), 2)
            self.upper = round(np.max(self.diff_std), 2)
            self.legend.updateBounds(self.lower, self.upper)
            self.legend.updateCriticalPoint(np.mean(self.diff_std))
            self.draw()
        elif self.binary_radio.isChecked():
            self.value = self.diff > 3 * np.mean(self.diff_std)
            self.lower = 0
            self.upper = 1
            self.legend.updateBounds(False, True)
            self.legend.updateCriticalPoint(None)
            self.draw()

    def setData(self, spots, diff, diff_std):
        self.spots = spots
        self.diff = diff
        self.diff_std = diff_std
        self.min = np.min(diff)
        self.max = np.max(diff)
        self.lower = self.min
        self.upper = self.max
        self.legend.updateBounds(round(self.min, 2), round(self.max, 2))
        crit = 3 * np.mean(self.diff_std)
        self.legend.updateCriticalPoint(crit)

        self.value = self.diff

    def draw(self):
        # Create a QPainter to draw on the pixmap
        painted_pixmap = self.pixmap.copy()
        painter = QPainter(painted_pixmap)
        pen = QPen(Qt.NoPen)  # No outline for the circles
        painter.setPen(pen)

        # Draw red filled circles at the spot positions
        for spot, value in zip(self.spots, self.value):
            x, y, r = spot
            # Normalize diff to [0,1] depending on min and max
            value = (value - self.lower) / (self.upper - self.lower)
            rgba = cmap(value)
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

    def updateBounds(self, lower, upper):
        self.lower = lower
        self.upper = upper
        self.draw()
