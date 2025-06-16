import pyqtgraph as pg
import numpy as np
import os
import csv

from PySide6.QtGui import QColor

from PySide6.QtWidgets import (
    QHBoxLayout,
    QVBoxLayout,
    QSpinBox,
    QLabel,
    QFileDialog,
    QGroupBox,
    QPushButton,
    QSizePolicy,
)

from misc.colors import color_palette, time_color_palette


class TimeSeries:
    def __init__(self, parent=None):
        self.parent = parent
        self.widget = pg.PlotWidget()
        # self.widget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.widget.setBackground("w")
        self.widget.setTitle("Sensorgram", color="black")
        self.widget.setLabel("left", "Extracted Wavelength", units="nm")
        self.widget.setLabel("bottom", "Frame Index")
        self.x_values = None
        self.y_values = None
        self.time_indices = None
        self.time_labels = None
        self.group_labels = None
        self.time_ranges = []
        self.time_range_spinboxes = []
        self.time_range_sb_layouts = []
        self.time_ranges_added = []
        self.curves = None
        self.time_controls_group_box = QGroupBox("Time Range Selection")

        # Layout the spinboxes and add time range button in UI
        self.spinbox_layout = QVBoxLayout()
        add_time_range_button = QPushButton("Add Time Range")
        add_time_range_button.clicked.connect(self.add_new_time_region)
        self.spinbox_layout.addWidget(add_time_range_button)

        self.time_controls_group_box.setLayout(self.spinbox_layout)

        # Add two time ranges for comparisons
        self.add_time_region()
        self.add_time_region()

    def reset_added(self):
        for i in range(len(self.time_ranges)):
            self.time_ranges_added[i] = False

    def add_time_region(self):
        index = len(self.time_ranges)
        region = pg.LinearRegionItem(
            orientation="vertical",
            movable=True,
            brush=QColor(*time_color_palette[index % len(time_color_palette)], 15),
        )
        self.time_ranges.append(region)

        # Create spinboxes
        start_spinbox = QSpinBox()
        end_spinbox = QSpinBox()
        start_spinbox.setValue(-1)

        self.time_range_spinboxes.append((start_spinbox, end_spinbox))

        layout = QHBoxLayout()

        layout.addWidget(QLabel(f"Range {index}:"))
        layout.addWidget(start_spinbox)
        layout.addWidget(end_spinbox)
        self.time_range_sb_layouts.append(layout)
        self.spinbox_layout.addLayout(layout)

        # Connect spinbox -> region
        start_spinbox.valueChanged.connect(
            lambda: region.setRegion((start_spinbox.value(), end_spinbox.value()))
        )
        end_spinbox.valueChanged.connect(
            lambda: region.setRegion((start_spinbox.value(), end_spinbox.value()))
        )
        region.sigRegionChanged.connect(self.update_spinboxes)
        self.time_ranges_added.append(False)
        if index > 1:
            self.parent.extinction_ui.updateTimeRanges()
            self.parent.extinction_ui.draw()
            self.parent.statistics_view.updateMeans()

    def add_new_time_region(self):
        self.add_time_region()
        self.addTimeRanges()

    def update_spinboxes(self):
        for range, (start_sb, end_sb) in zip(
            self.time_ranges, self.time_range_spinboxes
        ):
            r = range.getRegion()
            start_sb.blockSignals(True)
            end_sb.blockSignals(True)
            start_sb.setValue(r[0])
            end_sb.setValue(r[1])
            start_sb.blockSignals(False)
            end_sb.blockSignals(False)

        self.parent.extinction_ui.updateTimeRanges()
        self.parent.extinction_ui.updateCurvesData()
        self.parent.statistics_view.updateMeans()

    def cleanup(self):
        """
        Clean up the TimeSeries instance to prevent memory leaks.
        """
        if self.widget is not None:
            self.widget.clear()  # Clear any plots in the PlotWidget
            self.widget.deleteLater()  # Schedule the widget for deletion

        # Break references to instance variables
        self.x_values = None
        self.y_values = None
        self.time_indices = None
        self.time_labels = None
        self.group_labels = None

        # Optionally break parent reference to avoid circular references
        self.parent = None

    def updateTimeIndices(self, time_indices, time_labels):
        self.time_indices = time_indices
        self.time_labels = time_labels
        y_values = self.y_values[self.time_indices]
        x_values = self.x_values[self.time_indices]
        for i, col in enumerate((y_values.T)):
            self.curves[i].setData(x_values, col)

    def updateCurveData(self):
        self.x_values, self.y_values = self.parent.extinction_ui.get_time_series()
        x_values = self.x_values[self.time_indices]
        y_values = self.y_values[self.time_indices]
        for i, col in enumerate((y_values.T)):
            group = self.group_labels[i]
            spot_color = color_palette[group]
            pen = pg.mkPen(color=QColor(*spot_color), width=3)
            self.curves[i].setData(x_values, col)
            self.curves[i].setVisible(True)
            self.curves[i].setPen(pen)

    def draw(self):
        self.widget.clear()

        self.x_values, self.y_values = self.parent.extinction_ui.get_time_series()
        if self.time_indices is None:
            self.time_indices = np.array(range(len(self.x_values)))
            self.time_labels = np.zeros_like(self.time_indices)
        # TODO set maximum to two times spot count. Don't know if more would be necessary, check that
        self.max_num_groups = 2 * self.parent.extinction_ui.get_spot_count()
        self.curves = np.empty(self.max_num_groups, dtype=object)
        # Retrieve time series data from extinction curves

        # Plot lines and color according to spot label
        y_values = self.y_values[self.time_indices]
        x_values = self.x_values[self.time_indices]
        self.group_labels = self.parent.extinction_ui.get_groups()
        for i, col in enumerate((y_values.T)):
            group = self.group_labels[i]
            spot_color = color_palette[group]
            pen = pg.mkPen(color=QColor(*spot_color), width=3)
            line = pg.PlotDataItem(x_values, col, pen=pen, symbol=None)
            self.widget.addItem(line)
            self.curves[i] = line
        self.addTimeRanges()

    def addTimeRanges(self):
        for i, (range, (start_sb, end_sb), added) in enumerate(
            zip(self.time_ranges, self.time_range_spinboxes, self.time_ranges_added)
        ):
            if not added:
                self.widget.addItem(range)
                self.time_ranges_added[i] = True

                # If this is the inital setup:
                if start_sb == -1:
                    # Set time ranges to 1/10th of the total range
                    start = self.min_x
                    end = self.min_x + (self.max_x - self.min_x) / 10

                    range.setRegion((start, end))
                    start_sb.setValue(start)
                    end_sb.setValue(end)

                    # Set spinbox properties
                    for sb in [start_sb, end_sb]:
                        sb.setRange(self.min_x, self.max_x)
                        sb.setSingleStep(1)

    def export_time_series_to_csv(self, filename):
        # Prepare the data
        x_values = self.x_values[self.time_indices]
        y_values = self.y_values[self.time_indices]
        group_labels = self.group_labels

        # Write to CSV
        with open(filename, "w", newline="") as csvfile:
            csvwriter = csv.writer(csvfile)
            # Write header
            header = ["Group"] + [f"Timestep_{timestep:.2f}" for timestep in x_values]
            csvwriter.writerow(header)
            # Write data
            for i, label in enumerate(group_labels):
                row = [label] + y_values[:, i].tolist()
                csvwriter.writerow(row)

    def export(self):
        proposed_filename = os.path.join(os.curdir, "time_series_values.csv")
        filename, _ = QFileDialog.getSaveFileName(
            self.widget,
            "Export Time Series Values",
            proposed_filename,
            "CSV Files (*.csv)",
        )
        if filename:
            self.export_time_series_to_csv(filename)
