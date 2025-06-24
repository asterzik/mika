from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout, QFrame, QHBoxLayout
from misc.colors import color_palette, time_color_palette
from PySide6.QtGui import QColor
from math import sqrt
import numpy as np

import decimal


def format_with_aligned_decimals(val, std):
    std_str_g = f"{std:.3g}"  # Format std to 3 significant digits first

    # Determine the number of decimal places in std_str_g
    if "." in std_str_g:
        decimal_part = std_str_g.split(".")[1]
        # Exclude 'e' and its following digits if scientific notation is used
        if "e" in decimal_part:
            decimal_part = decimal_part.split("e")[0]
        num_decimal_places = len(decimal_part)
    else:
        num_decimal_places = 0

    # Format val to that many decimal places (fixed-point notation)
    val_str = f"{val:.{num_decimal_places}f}"
    return std_str_g, val_str


class StatisticsFrame(QFrame):
    def __init__(self, parent, group_label, group_name, means, stds):
        super().__init__()
        self.parent = parent
        self.group_label = group_label
        self.group_name = group_name
        self.means = means
        self.stds = stds
        self.sig_digits = 3
        self.diff_labels = []
        self.diff = np.zeros(len(self.means) - 1)
        self.diff_std = np.zeros(len(self.stds) - 1)
        for i, mean in enumerate(self.means):
            if i == 0:
                continue
            self.diff[i - 1] = mean - self.means[0]
        for i, std in enumerate(self.stds):
            if i == 0:
                continue
            self.diff_std[i - 1] = sqrt(std**2 + self.stds[0] ** 2)

        rgb_color = color_palette[self.group_label]
        self.color_str = f"rgb({rgb_color[0]}, {rgb_color[1]}, {rgb_color[2]})"

        self.setFrameShape(QFrame.Box)
        self.setStyleSheet(
            f"""
            QFrame {{
                background-color: #f4f4f4;
                border: 2px solid {self.color_str}; 
                border-radius: 4px;
                padding: 5px;
            }}
            QLabel {{
                font-size: 12px;
                border: 1px solid gray;
            }}
        """
        )
        self.layout = QHBoxLayout()
        self.layout.setContentsMargins(8, 2, 8, 2)  # Minimize inner padding
        self.layout.setSpacing(15)  # Space between labels

        title_label = QLabel(f"<b>{self.group_name}</b>")
        title_label.setFixedWidth(80)  # Align group names

        self.layout.addWidget(title_label)
        self.setLayout(self.layout)
        self.addDiffLabel()
        # self.layout.addStretch()  # Push everything neatly to the left

        # Check if the difference is larger than 3 times the standard deviation
        # check_label = QLabel()
        # check_label.setText("✔️")  # Unicode checkmark
        # if abs(self.diff) < 3 * self.diff_std:
        #     check_label.setVisible(False)

        # Store references to the QLabel widgets

    def addDiffLabel(self):
        index = len(self.diff_labels)
        val = self.diff[index]
        std = self.diff_std[index]

        std_str, val_str = format_with_aligned_decimals(val, std)

        diff_label = QLabel(f"<b>Diff {index + 1}:</b> {val_str}±{std_str}")
        self.layout.addWidget(diff_label)
        self.diff_labels.append(diff_label)

    def updateData(self, means, stds):
        self.means = means
        self.stds = stds
        self.diff = np.zeros(len(self.means) - 1)
        self.diff_std = np.zeros(len(self.stds) - 1)
        for i, mean in enumerate(self.means):
            if i == 0:
                continue
            self.diff[i - 1] = mean - self.means[0]
        for i, std in enumerate(self.stds):
            if i == 0:
                continue
            self.diff_std[i - 1] = sqrt(std**2 + self.stds[0] ** 2)

    def updateDiffLabels(self):
        for i, diff_label in enumerate(self.diff_labels):
            std_str, val_str = format_with_aligned_decimals(
                self.diff[i], self.diff_std[i]
            )
            diff_label.setText(f"<b>Diff {i + 1}:</b> {val_str}±{std_str}")

    def updateDiffColors(self):
        for i, diff_label in enumerate(self.diff_labels):
            color = (int)((i + 1) * (255 / (len(self.diff_labels))))
            diff_label.setStyleSheet(
                f"border: 2px solid rgb({color}, {color}, {color})"
            )

    def clear(self):
        while self.layout.count():
            item = self.layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()


class StatisticsView(QWidget):
    def __init__(self, parent, title="Means and Standard Errors"):
        super().__init__()
        self.title = title
        self.parent = parent

        self.layout = QVBoxLayout()
        self.layout.setSpacing(5)  # Reduce spacing between rows
        # Section Title Label (inside the widget)

        self.setLayout(self.layout)
        self.frames = []

    def clear(self):
        while self.layout.count():
            item = self.layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)  # Remove from layout
                widget.deleteLater()  # Schedule deletion
        for frame in self.frames:
            frame.clear()
        self.frames = []

    def init_groups(self, means, stds, group_labels):
        self.setUpdatesEnabled(False)
        self.clear()

        title_label = QLabel(self.title)
        self.layout.addWidget(title_label)
        for i, label in enumerate(group_labels):
            group_name = self.parent.spot_ui.group_names[label]
            mean = means[:, i]
            std = stds[:, i]
            frame = StatisticsFrame(self.parent, label, group_name, mean, std)
            self.frames.append(frame)
            self.layout.addWidget(frame)
        self.layout.addStretch()

        self.setUpdatesEnabled(True)
        self.update()

    def updateMeans(self):
        means, stds = self.parent.extinction_ui.get_statistics()
        for index, frame in enumerate(self.frames):
            mean = means[:, index]
            std = stds[:, index]
            to_add = len(mean) - len(frame.means)
            frame.updateData(mean, std)
            for i in range(to_add):
                frame.addDiffLabel()
            frame.updateDiffLabels()

    def getData(self):
        return self.diff, self.diff_std

    def updateDiffColors(self):
        for index, frame in enumerate(self.frames):
            frame.updateDiffColors()
