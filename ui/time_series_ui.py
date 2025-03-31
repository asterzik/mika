import pyqtgraph as pg
import numpy as np
import os
import csv

from PySide6.QtGui import QColor

from PySide6.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QPushButton,
    QRadioButton,
    QButtonGroup,
    QSpacerItem,
    QSizePolicy,
    QFileDialog,
)

from misc.colors import color_palette, time_color_palette


class TimeSeries:
    def __init__(self, parent=None):
        self.parent = parent
        self.widget = pg.PlotWidget()
        self.widget.setBackground("w")
        self.widget.setTitle("Time Series", color="black")
        self.x_values = None
        self.y_values = None
        self.time_indices = None
        self.time_labels = None
        self.group_labels = None
        self.curves = None

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
        self.max_num_groups = self.parent.extinction_ui.get_spot_count()
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
        # Add two time ranges for comparisons

        color1 = QColor()
        color1.setRgb(*time_color_palette[0])
        color1.setAlpha(15)

        # Create a new LinearRegionItem (the time range)
        self.time_region1 = pg.LinearRegionItem(
            orientation="vertical", movable=True, brush=color1
        )
        min_x = np.min(self.x_values)
        max_x = np.max(self.x_values)
        self.time_region1.setRegion((min_x, np.floor(max_x / 2)))
        self.widget.addItem(self.time_region1)

        def on_time_region_change():
            self.parent.extinction_ui.updateTimeRanges()
            self.parent.extinction_ui.updateCurvesData()
            self.parent.statistics_view.updateMeans()

        self.time_region1.sigRegionChangeFinished.connect(on_time_region_change)

        color2 = QColor()
        color2.setRgb(*time_color_palette[1])
        color2.setAlpha(15)

        # Create a new LinearRegionItem (the time range)
        self.time_region2 = pg.LinearRegionItem(
            orientation="vertical", movable=True, brush=color2
        )
        self.time_region2.setRegion((np.ceil((max_x / 2)), max_x))

        self.time_region2.sigRegionChangeFinished.connect(on_time_region_change)
        self.widget.addItem(self.time_region2)

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
