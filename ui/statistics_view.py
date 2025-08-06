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
        self.mean_labels = []  # New: to store mean labels

        # Calculate diffs and diff_stds
        self.diff = np.zeros(len(self.means) - 1)
        self.diff_std = np.zeros(len(self.stds) - 1)
        for i in range(1, len(self.means)):
            self.diff[i - 1] = self.means[i] - self.means[0]
            self.diff_std[i - 1] = sqrt(self.stds[i] ** 2 + self.stds[0] ** 2)

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
                padding: 2px 5px; /* Add some padding to labels */
            }}
        """
        )

        # Main layout for the frame (vertical to hold two lines)
        self.main_layout = QVBoxLayout()
        self.main_layout.setContentsMargins(8, 2, 8, 2)
        self.main_layout.setSpacing(5)  # Space between the two lines

        # Layout for the Diffs line
        self.diff_line_layout = QHBoxLayout()
        self.diff_line_layout.setContentsMargins(
            0, 0, 0, 0
        )  # No extra margins for sub-layouts
        self.diff_line_layout.setSpacing(15)

        # Layout for the Means line
        self.mean_line_layout = QHBoxLayout()
        self.mean_line_layout.setContentsMargins(
            0, 0, 0, 0
        )  # No extra margins for sub-layouts
        self.mean_line_layout.setSpacing(15)

        # Add the group title to the diff line
        self.title_label = QLabel(f"<b>{self.group_name}</b>")
        self.title_label.setFixedWidth(80)  # Align group names
        self.diff_line_layout.addWidget(self.title_label)

        self.addStatisticsLabels()  # Now adds both diffs and means

        # Add stretch to push labels to the left
        self.diff_line_layout.addStretch()
        self.mean_line_layout.addStretch()

        # Add the two line layouts to the main layout
        self.main_layout.addLayout(self.diff_line_layout)
        self.main_layout.addLayout(self.mean_line_layout)

        self.setLayout(self.main_layout)

    def addStatisticsLabels(self):
        # Clear existing labels before adding new ones (useful for updates)
        self._clear_labels(self.diff_labels, self.diff_line_layout)
        self._clear_labels(self.mean_labels, self.mean_line_layout)

        # Add Mean Labels
        for i, mean_val in enumerate(self.means):
            mean_std = self.stds[i]
            mean_std_str, mean_val_str = format_with_aligned_decimals(
                mean_val, mean_std
            )

            if i == 0:
                label_text = f"<b>Reference:</b> {mean_val_str}±{mean_std_str}"
            else:
                label_text = f"<b>Mean {i}:</b> {mean_val_str}±{mean_std_str}"

            mean_label = QLabel(label_text)
            self.mean_line_layout.addWidget(mean_label)
            self.mean_labels.append(mean_label)

        # Add Diff Labels
        for i in range(len(self.diff)):
            val = self.diff[i]
            std = self.diff_std[i]

            std_str, val_str = format_with_aligned_decimals(val, std)

            diff_label = QLabel(f"<b>Diff {i + 1}:</b> {val_str}±{std_str}")
            self.diff_line_layout.addWidget(diff_label)
            self.diff_labels.append(diff_label)

        self.diff_line_layout.addStretch()  # Add stretch to push labels to the left
        self.mean_line_layout.addStretch()  # Add stretch to push labels to the left

    def _clear_labels(self, label_list, layout):
        # Iterate backwards because we'll be removing items as we go
        for i in reversed(range(layout.count())):
            item = layout.itemAt(i)
            if item.spacerItem():  # This checks if it's a stretch/spacer
                layout.removeItem(item)

        # Remove and delete all QLabel widgets from a list and its layout
        for label in label_list:
            if label:
                layout.removeWidget(label)
                label.setParent(None)
                label.deleteLater()
        label_list.clear()  # Clear the list itself

    def updateData(self, means, stds):
        self.means = means
        self.stds = stds
        self.diff = np.zeros(len(self.means) - 1)
        self.diff_std = np.zeros(len(self.stds) - 1)
        for i in range(1, len(self.means)):
            self.diff[i - 1] = self.means[i] - self.means[0]
            self.diff_std[i - 1] = sqrt(self.stds[i] ** 2 + self.stds[0] ** 2)

        self.updateStatisticsLabels()  # Call a combined update method

    def updateStatisticsLabels(self):
        # Update Mean Labels
        for i, mean_label in enumerate(self.mean_labels):
            mean_val = self.means[i]
            mean_std = self.stds[i]
            mean_std_str, mean_val_str = format_with_aligned_decimals(
                mean_val, mean_std
            )
            if i == 0:
                mean_label.setText(f"<b>Reference:</b> {mean_val_str}±{mean_std_str}")
            else:
                mean_label.setText(f"<b>Mean {i}:</b> {mean_val_str}±{mean_std_str}")

        # Update Diff Labels
        for i, diff_label in enumerate(self.diff_labels):
            std_str, val_str = format_with_aligned_decimals(
                self.diff[i], self.diff_std[i]
            )
            diff_label.setText(f"<b>Diff {i + 1}:</b> {val_str}±{std_str}")

    def updateDiffColors(self):
        # This function currently only updates diff colors.
        # Consider if you also want to update mean label colors.
        for i, diff_label in enumerate(self.diff_labels):
            # Example color calculation, you might want to adjust this
            color_val = (int)(
                (i + 1) * (255 / (len(self.diff_labels) + 1))
            )  # +1 to avoid full 255 for last one if only one diff
            diff_label.setStyleSheet(
                f"border: 2px solid rgb({color_val}, {color_val}, {color_val}); font-size: 12px; padding: 2px 5px;"
            )
        # You might want to add similar logic for mean_labels if needed
        # for i, mean_label in enumerate(self.mean_labels):
        #     mean_label.setStyleSheet(...)

    def clear(self):
        # Clear both lines of labels
        self._clear_labels(self.diff_labels, self.diff_line_layout)
        self._clear_labels(self.mean_labels, self.mean_line_layout)

        # Clear the main layouts content, excluding the fixed title/group1 labels
        # This will remove the spacers as well
        for i in reversed(range(self.diff_line_layout.count())):
            item = self.diff_line_layout.itemAt(i)
            if item.widget() and item.widget() not in [
                self.title_label
            ]:  # Don't remove title label
                widget = item.widget()
                self.diff_line_layout.removeWidget(widget)
                widget.setParent(None)
                widget.deleteLater()
            elif item.spacerItem():
                self.diff_line_layout.removeItem(item)

        for i in reversed(range(self.mean_line_layout.count())):
            item = self.mean_line_layout.itemAt(i)
            if item.widget() and item.widget() not in [
                self.mean_line_layout.itemAt(0).widget()
            ]:  # Don't remove Group1 label
                widget = item.widget()
                self.mean_line_layout.removeWidget(widget)
                widget.setParent(None)
                widget.deleteLater()
            elif item.spacerItem():
                self.mean_line_layout.removeItem(item)

        # Re-add stretches after clearing if they were removed
        # self.diff_line_layout.addStretch()
        # self.mean_line_layout.addStretch()


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
                frame.addStatisticsLabels()
            frame.updateStatisticsLabels()

    def getData(self):
        return self.diff, self.diff_std

    def updateDiffColors(self):
        for index, frame in enumerate(self.frames):
            frame.updateDiffColors()
