from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout, QFrame, QHBoxLayout
from misc.colors import color_palette, time_color_palette
from PySide6.QtGui import QColor
from math import sqrt


class StatisticsView(QWidget):
    def __init__(self, parent, title="Means and Standard Errors"):
        super().__init__()
        self.title = title
        self.parent = parent

        self.layout = QVBoxLayout()
        self.layout.setSpacing(5)  # Reduce spacing between rows
        # Section Title Label (inside the widget)

        self.setLayout(self.layout)
        self.label_references = {}  # Dictionary to store references to QLabel widgets

    def clear(self):
        while self.layout.count():
            item = self.layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)  # Remove from layout
                widget.deleteLater()  # Schedule deletion
        self.label_references.clear()

    def init_groups(self, means, stds, group_labels):
        self.setUpdatesEnabled(False)
        self.clear()

        title_label = QLabel(self.title)
        self.layout.addWidget(title_label)
        for i, label in enumerate(group_labels):
            group_name = f"Group {label+1}"
            group_name = self.parent.spot_ui.group_names[label]
            mean = means[:, i]
            std = stds[:, i]
            frame = self.create_stat_frame(group_name, label, mean, std, i)
            self.layout.addWidget(frame)
        self.layout.addStretch()

        self.setUpdatesEnabled(True)
        self.update()

    def create_stat_frame(self, group_name, group_label, mean, std, index):
        """Creates a single-line frame with statistics for a group."""

        rgb_color = color_palette[group_label]
        color_str = f"rgb({rgb_color[0]}, {rgb_color[1]}, {rgb_color[2]})"

        frame = QFrame()
        frame.setFrameShape(QFrame.Box)
        frame.setStyleSheet(
            f"""
            QFrame {{
                background-color: #f4f4f4;
                border: 2px solid {color_str}; 
                border-radius: 4px;
                padding: 5px;
            }}
            QLabel {{
                font-size: 12px;
                border: 1px solid gray;
            }}
        """
        )

        layout = QHBoxLayout()
        layout.setContentsMargins(8, 2, 8, 2)  # Minimize inner padding
        layout.setSpacing(15)  # Space between labels

        title_label = QLabel(f"<b>{group_name}</b>")
        title_label.setFixedWidth(80)  # Align group names

        mean_label = QLabel(f"{mean[0]:.3f}±{std[0]:.3f}")
        mean_range1_label = QLabel(f"{mean[1]:.3f}±{std[1]:.3f}")
        mean_range2_label = QLabel(f"{mean[2]:.3f}±{std[2]:.3f}")
        self.diff = mean[2] - mean[1]
        self.diff_std = sqrt(std[1] * std[1] + std[2] * std[2])
        diff_label = QLabel(f"<b>Diff R2 - R1:</b> {self.diff:.3f}±{self.diff_std:.3f}")

        # Check if the difference is larger than 3 times the standard deviation
        check_label = QLabel()
        check_label.setText("✔️")  # Unicode checkmark
        if abs(self.diff) < 3 * self.diff_std:
            check_label.setVisible(False)

        # Store references to the QLabel widgets
        self.label_references[index] = {
            "mean_label": mean_label,
            "mean_range1_label": mean_range1_label,
            "mean_range2_label": mean_range2_label,
            "diff_label": diff_label,
            "check_label": check_label,
        }

        # Color the ranges widgets
        mean_range1_color = time_color_palette[0]
        mean_range1_label.setStyleSheet(
            f"""
            background-color: rgba({mean_range1_color[0]}, {mean_range1_color[1]}, {mean_range1_color[2]}, 30);
            border: 3px solid {color_str};
            border-style: dashed;
            """
        )

        mean_range2_color = time_color_palette[1]
        mean_range2_label.setStyleSheet(
            f"""
            background-color: rgba({mean_range2_color[0]}, {mean_range2_color[1]}, {mean_range2_color[2]}, 30);
            border: 3px solid {color_str};
            """
        )

        layout.addWidget(title_label)
        # layout.addWidget(mean_label)
        layout.addWidget(mean_range1_label)
        layout.addWidget(mean_range2_label)
        layout.addWidget(diff_label)
        layout.addWidget(check_label)
        layout.addStretch()  # Push everything neatly to the left

        frame.setLayout(layout)
        return frame

    def updateMeans(self):
        means, stds = self.parent.extinction_ui.get_statistics()
        for index, labels in self.label_references.items():
            mean = means[:, index]
            std = stds[:, index]
            # labels["mean_label"].setText(f"{mean[0]:.3f}±{std[0]:.3f}")
            labels["mean_range1_label"].setText(f"{mean[1]:.3f}±{std[1]:.3f}")
            labels["mean_range2_label"].setText(f"{mean[2]:.3f}±{std[2]:.3f}")
            self.diff = mean[2] - mean[1]
            self.diff_std = sqrt(std[1] * std[1] + std[2] * std[2])
            labels["diff_label"].setText(
                f"<b>Diff R2 - R1:</b> {self.diff:.3f}±{self.diff_std:.3f}"
            )
            if abs(self.diff) > 3 * self.diff_std:
                labels["check_label"].setVisible(True)
            else:
                labels["check_label"].setVisible(False)

    def getData(self):
        return self.diff, self.diff_std
