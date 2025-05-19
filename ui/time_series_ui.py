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
        self.curves = None
        self.time_controls_group_box = QGroupBox("Time Range Selection")

        # Add two time ranges for comparisons

        color1 = QColor()
        color1.setRgb(*time_color_palette[0])
        color1.setAlpha(15)

        color2 = QColor()
        color2.setRgb(*time_color_palette[1])
        color2.setAlpha(15)

        self.time_region1 = pg.LinearRegionItem(
            orientation="vertical", movable=True, brush=color1
        )
        self.time_region2 = pg.LinearRegionItem(
            orientation="vertical", movable=True, brush=color2
        )

        # Create spinboxes
        self.spinbox_region1_start = QSpinBox()
        self.spinbox_region1_end = QSpinBox()
        self.spinbox_region2_start = QSpinBox()
        self.spinbox_region2_end = QSpinBox()

        # Layout the spinboxes somewhere in your UI
        spinbox_layout = QVBoxLayout()
        s1_layout = QHBoxLayout()
        s2_layout = QHBoxLayout()
        s1_layout.addWidget(QLabel("Range 1:"))
        s1_layout.addWidget(self.spinbox_region1_start)
        s1_layout.addWidget(self.spinbox_region1_end)
        s2_layout.addWidget(QLabel("Range 2:"))
        s2_layout.addWidget(self.spinbox_region2_start)
        s2_layout.addWidget(self.spinbox_region2_end)
        spinbox_layout.addLayout(s1_layout)
        spinbox_layout.addLayout(s2_layout)

        self.time_controls_group_box.setLayout(spinbox_layout)
        # self.parent.layout().addWidget(self.time_controls_widget)  # Add to your UI layout

        # Connect spinbox -> region
        self.spinbox_region1_start.valueChanged.connect(
            lambda: self.time_region1.setRegion(
                (self.spinbox_region1_start.value(), self.spinbox_region1_end.value())
            )
        )
        self.spinbox_region1_end.valueChanged.connect(
            lambda: self.time_region1.setRegion(
                (self.spinbox_region1_start.value(), self.spinbox_region1_end.value())
            )
        )
        self.spinbox_region2_start.valueChanged.connect(
            lambda: self.time_region2.setRegion(
                (self.spinbox_region2_start.value(), self.spinbox_region2_end.value())
            )
        )
        self.spinbox_region2_end.valueChanged.connect(
            lambda: self.time_region2.setRegion(
                (self.spinbox_region2_start.value(), self.spinbox_region2_end.value())
            )
        )

        # Connect region -> spinbox

        self.time_region1.sigRegionChanged.connect(self.update_spinboxes)
        self.time_region2.sigRegionChanged.connect(self.update_spinboxes)

    def update_spinboxes(self):
        r1 = self.time_region1.getRegion()
        r2 = self.time_region2.getRegion()
        self.spinbox_region1_start.blockSignals(True)
        self.spinbox_region1_end.blockSignals(True)
        self.spinbox_region2_start.blockSignals(True)
        self.spinbox_region2_end.blockSignals(True)
        self.spinbox_region1_start.setValue(r1[0])
        self.spinbox_region1_end.setValue(r1[1])
        self.spinbox_region2_start.setValue(r2[0])
        self.spinbox_region2_end.setValue(r2[1])
        self.spinbox_region1_start.blockSignals(False)
        self.spinbox_region1_end.blockSignals(False)
        self.spinbox_region2_start.blockSignals(False)
        self.spinbox_region2_end.blockSignals(False)

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
        self.widget.addItem(self.time_region1)
        self.widget.addItem(self.time_region2)
        # Store initial values
        self.min_x = np.min(self.x_values)
        self.max_x = np.max(self.x_values)
        mid_x = (self.min_x + self.max_x) / 2
        print(self.min_x, self.max_x, mid_x)

        # Set initial regions
        self.time_region1.setRegion((self.min_x, mid_x))
        self.time_region2.setRegion((mid_x, self.max_x))

        for sb in [
            self.spinbox_region1_start,
            self.spinbox_region1_end,
            self.spinbox_region2_start,
            self.spinbox_region2_end,
        ]:
            sb.setRange(self.min_x, self.max_x)
            sb.setSingleStep(1)

        # Set initial values
        self.spinbox_region1_start.setValue(self.min_x)
        self.spinbox_region1_end.setValue(mid_x)
        self.spinbox_region2_start.setValue(mid_x)
        self.spinbox_region2_end.setValue(self.max_x)

        # self.time_region1.setRegion((min_x, np.floor(max_x / 2)))
        # self.widget.addItem(self.time_region1)

        # def on_time_region_change():
        #     self.parent.extinction_ui.updateTimeRanges()
        #     self.parent.extinction_ui.updateCurvesData()
        #     self.parent.statistics_view.updateMeans()

        # self.time_region1.sigRegionChangeFinished.connect(on_time_region_change)

        # self.time_region2.setRegion((np.ceil((max_x / 2)), max_x))
        # min_x = np.min(self.x_values)
        # max_x = np.max(self.x_values)

        # self.time_region2.sigRegionChangeFinished.connect(on_time_region_change)
        # self.widget.addItem(self.time_region2)

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
