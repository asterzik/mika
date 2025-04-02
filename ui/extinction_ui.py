from PySide6.QtWidgets import (
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
    QRadioButton,
    QGroupBox,
    QFormLayout,
    QCheckBox,
    QFileDialog,
)
from PySide6.QtGui import QColor
import pyqtgraph as pg
import numpy as np
import csv
import os

from enum import Enum

from line_fitting.line_fitting import Polynomial, GPRegression
from misc.colors import color_palette
from misc.profiling import ProfileContext
import multiprocessing as mp


def individualRegression(f, i, wavelengths, extinction, gp_bool, poly_degree):
    x = wavelengths[f]
    y = extinction[f, :, i]

    if gp_bool:
        regressor = GPRegression(x, y)
    else:
        regressor = Polynomial(x, y, poly_degree)

    regressor.fit()

    return regressor


class ExtinctionUi:
    def __init__(self, parent=None):
        self.parent = parent
        self.widget = QWidget()
        self.layout = QHBoxLayout()
        self.plot_widget = pg.PlotWidget()
        self.layout.addWidget(self.plot_widget)
        self.button_layout = QVBoxLayout()
        self.selection_layout = QVBoxLayout()
        self.layout.addLayout(self.selection_layout)
        self.extinction_display_options()
        self.group_averaging_options()
        self.plot_widget.setBackground("w")
        self.plot_widget.setTitle("Extinction", color="black")
        self.spot_indices = None
        self.spot_labels = None
        self.time_indices = None
        self.time_labels = None
        self.poly_degree = 5
        self.regressors = None
        self.num_time_steps = None
        self.num_spots = None
        self.curves = None
        self.num_regression_points = 40
        self.time_range1_indices = None
        self.time_range2_indices = None
        self.metric = None
        self.time_average_curves = None

        self.regressor_selection_ui()
        self.metric_selection_ui()

        self.widget.setLayout(self.layout)

    def cleanup(self):
        """Clean up resources held by this object."""

        if self.regressors is not None:
            for row in range(self.regressors.shape[0]):
                for col in range(self.regressors.shape[1]):
                    regressor = self.regressors[row, col]
                    if regressor is not None:
                        regressor.cleanup()  # Call the cleanup method of each regressor

        self.regressors = None

        # Clean up plot_widget
        if self.plot_widget is not None:
            self.plot_widget.clear()  # Clear any data in the plot
            self.plot_widget.deleteLater()  # Schedule it for deletion
            self.plot_widget = None

        # Clean up the main widget and layout
        if self.widget is not None:
            self.widget.deleteLater()  # Safely delete the widget
            self.widget = None

        # Optionally: Break the reference to the parent
        self.parent = None

        # Set other attributes to None (if necessary)
        self.spot_indices = None
        self.spot_labels = None
        self.time_indices = None
        self.time_labels = None
        self.poly_degree = None

    def computeEverything(self):
        self.computeExtinction()
        self.setAllIndices()
        if self.curves is None:
            self.curves = np.empty((self.num_time_steps, self.num_spots), dtype=object)
        self.generate_group_lists()
        if self.average_first_radio.isChecked():
            self.averageGroups()
        self.regression()

    def extinction_display_options(self):
        group_box = QGroupBox("Extinction Display")
        group_layout = QVBoxLayout()
        group_box.setLayout(group_layout)
        self.selection_layout.addWidget(group_box)

        self.raw_data_checkbox = QCheckBox("Raw Data")
        self.raw_data_checkbox.setChecked(False)
        group_layout.addWidget(self.raw_data_checkbox)
        self.raw_data_checkbox.toggled.connect(self.toggle_raw_data)

        self.maxima_checkbox = QCheckBox("Chosen curve metric")
        self.maxima_checkbox.setChecked(False)
        group_layout.addWidget(self.maxima_checkbox)
        self.maxima_checkbox.toggled.connect(self.toggle_maxima)

        self.average_time_checkbox = QCheckBox("Time average curves")
        self.average_time_checkbox.setChecked(True)
        group_layout.addWidget(self.average_time_checkbox)
        self.average_time_checkbox.toggled.connect(self.toggle_time_average)

    def group_averaging_options(self):
        group_box = QGroupBox("Group Averaging")
        group_layout = QVBoxLayout()
        group_box.setLayout(group_layout)
        self.selection_layout.addWidget(group_box)

        self.average_first_radio = QRadioButton("Average extinction data")
        self.average_later_radio = QRadioButton("Average calculated metric")
        self.average_first_radio.setChecked(True)

        group_layout.addWidget(self.average_first_radio)
        group_layout.addWidget(self.average_later_radio)

        self.average_first_radio.toggled.connect(self.updateAverageGroups)

    def updateAverageGroups(self):
        if self.average_first_radio.isChecked():
            self.averageGroups()
        self.regression(first=False)
        self.draw()
        self.parent.time_series.updateCurveData()

    def regressor_selection_ui(self):
        group_box = QGroupBox("Select Regressor")
        group_layout = QVBoxLayout()

        self.poly_radio = QRadioButton("Polynomial")
        self.gp_radio = QRadioButton("GP")

        self.poly_radio.setChecked(True)

        group_layout.addWidget(self.poly_radio)
        group_layout.addWidget(self.gp_radio)

        group_box.setLayout(group_layout)

        self.button_layout.addWidget(group_box)

        self.poly_radio.toggled.connect(self.updateRegressor)

    def metric_selection_ui(self):
        group_box = QGroupBox("Select Metric")
        group_layout = QVBoxLayout()

        self.max_radio = QRadioButton("Maximum")
        self.centroid_radio = QRadioButton("Centroid")
        self.centroid_left_bound_radio = QRadioButton("Centroid left bound")
        self.centroid_half_height_radio = QRadioButton("Centroid half height")
        self.centroid_left_bound_half_height_radio = QRadioButton(
            "Centroid left bound half height"
        )
        self.inflection_radio = QRadioButton("Inflection")
        # self.cross_correlation_radio = QRadioButton("Cross correlation")

        self.max_radio.setChecked(True)
        # self.cross_correlation_radio.setChecked(True)

        group_layout.addWidget(self.max_radio)
        group_layout.addWidget(self.centroid_radio)
        group_layout.addWidget(self.centroid_left_bound_radio)
        group_layout.addWidget(self.centroid_half_height_radio)
        group_layout.addWidget(self.centroid_left_bound_half_height_radio)
        group_layout.addWidget(self.inflection_radio)
        # group_layout.addWidget(self.cross_correlation_radio)

        group_box.setLayout(group_layout)

        self.button_layout.addWidget(group_box)

        self.max_radio.toggled[bool].connect(self.updateMetric)
        self.centroid_radio.toggled[bool].connect(self.updateMetric)
        self.centroid_left_bound_radio.toggled[bool].connect(self.updateMetric)
        self.centroid_half_height_radio.toggled[bool].connect(self.updateMetric)
        self.centroid_left_bound_half_height_radio.toggled[bool].connect(
            self.updateMetric
        )
        self.inflection_radio.toggled[bool].connect(self.updateMetric)
        # self.cross_correlation_radio.toggled[bool].connect(self.updateMetric)

    def updateRegressor(self):
        self.regression(first=False)
        self.updateCurvesData()
        self.parent.time_series.updateCurveData()

    def plot(self, time, time_label_index, spot_index, group_index, spot_label):
        regressor = self.regressors[time][
            spot_index
        ]  # spot_index = group_index for average first
        x_values, y_values = regressor.generateValues(self.num_regression_points)
        rgb_color = color_palette[spot_label]
        pen = pg.mkPen(color=QColor(*rgb_color), width=3)
        line = pg.PlotDataItem(x_values, y_values, pen=pen, symbol=None)
        self.plot_widget.addItem(line)
        self.curves[time][group_index] = line

    def compute_average_over_time(self, time_range_indices, group_index):
        sorted_wavelengths = np.sort(self.wavelengths[time_range_indices], axis=1)
        if self.average_first_radio.isChecked():

            sorted_extinction = np.zeros_like(
                self.average_extinction[time_range_indices, :, group_index]
            )
            for i, idx in enumerate(time_range_indices):
                sort_indices = np.argsort(self.wavelengths[idx])
                sorted_extinction[i] = self.average_extinction[
                    idx, sort_indices, group_index
                ]
        else:
            sorted_extinction = np.zeros_like(
                self.extinction[time_range_indices, :, group_index]
            )
            for i, idx in enumerate(time_range_indices):
                sort_indices = np.argsort(self.wavelengths[idx])
                sorted_extinction[i] = self.extinction[idx, sort_indices, group_index]

        # Compute mean values for plotting
        x_mean = np.mean(sorted_wavelengths, axis=0)
        y_mean = np.mean(sorted_extinction, axis=0)

        # Select regressor based on user choice
        if self.gp_radio.isChecked():
            regressor = GPRegression(x_mean, y_mean)
        else:
            regressor = Polynomial(x_mean, y_mean, self.poly_degree)

        # Fit regressor and generate plot data
        regressor.fit()
        return regressor.generateValues(self.num_regression_points)

    def plot_average_over_time(self, spot_index, group_index, spot_label):
        # Create and add plot line
        spot_color = color_palette[spot_label]
        pen1 = pg.mkPen(color=QColor(*spot_color), width=3, style=pg.QtCore.Qt.DashLine)
        pen2 = pg.mkPen(color=QColor(*spot_color), width=3)

        # Fit and plot for both time ranges

        x_plot, y_plot = self.compute_average_over_time(
            self.time_range1_indices, spot_index
        )
        line = pg.PlotDataItem(x_plot, y_plot, pen=pen1, symbol=None)
        self.plot_widget.addItem(line)
        self.time_average_curves[0, group_index] = line

        x_plot, y_plot = self.compute_average_over_time(
            self.time_range2_indices, spot_index
        )
        line = pg.PlotDataItem(x_plot, y_plot, pen=pen2, symbol=None)
        self.plot_widget.addItem(line)
        self.time_average_curves[1, group_index] = line

    def updateAverageOverTime(self, group_index):
        x, y = self.compute_average_over_time(self.time_range1_indices, group_index)
        self.time_average_curves[0, group_index].setData(x, y)

        x, y = self.compute_average_over_time(self.time_range2_indices, group_index)
        self.time_average_curves[1, group_index].setData(x, y)

    def toggle_time_average(self):
        self.draw()

    def updateCurveData(self, time, group_index):
        regressor = self.regressors[time][group_index]
        x_values, y_values = regressor.generateValues(self.num_regression_points)
        self.curves[time][group_index].setData(x_values, y_values)

    def setAllIndices(self):
        self.time_indices = np.array(range(self.num_time_steps))
        self.time_labels = np.zeros_like((self.time_indices))
        self.spot_indices = np.array(range(self.extinction[0].shape[1]))
        self.spot_labels = np.zeros_like(self.spot_indices)

    def computeExtinction(self):

        if (
            not self.parent.spot_ui.circles.extinction_bool
            and self.parent.spot_ui.circles.detected
        ):
            self.parent.spot_ui.circles.compute_extinction()
        if self.parent.spot_ui.circles.extinction_bool:
            frames, x_axis_values, extinction_values = (
                self.parent.spot_ui.circles.get_extinction()
            )

            # Sort all arrays
            sorted_frame_indices = np.argsort(frames)
            frames = frames[sorted_frame_indices]
            extinction_values = extinction_values[sorted_frame_indices]
            x_axis_values = x_axis_values[sorted_frame_indices]

            self.num_time_steps = len(np.unique(frames))
            self.wavelengths = len(frames) // self.num_time_steps
            self.selected_spot_indices = (
                self.parent.spot_ui.circles.getSelectedSpotIndices()
            )
            self.selected_spot_labels = (
                self.parent.spot_ui.circles.getSelectedSpotLabels()
            )
            self.num_spots = extinction_values.shape[1]

            self.extinction = extinction_values.reshape(
                (self.num_time_steps, self.wavelengths, self.num_spots)
            )
            self.wavelengths = x_axis_values.reshape(
                (self.num_time_steps, self.wavelengths)
            )

    def generate_group_lists(self):
        self.group_labels = np.unique(
            [label for sublist in self.selected_spot_labels for label in sublist]
        )
        self.num_groups = len(self.group_labels)

    def averageGroups(self):
        spot_indices = self.selected_spot_indices
        spot_labels = self.selected_spot_labels

        # Initialize arrays to store the sum and count of extinction values for each group
        sum_extinction_per_group = np.zeros(
            (self.extinction.shape[0], self.extinction.shape[1], self.num_groups)
        )
        count_per_group = np.zeros(self.num_groups)

        # Accumulate the sums and counts for each group
        for i, labels in enumerate(spot_labels):
            for label in labels:
                group_index = np.where(self.group_labels == label)[0][0]
                sum_extinction_per_group[:, :, group_index] += self.extinction[
                    :, :, spot_indices[i]
                ]
                count_per_group[group_index] += 1

        # Compute the average extinction for each group
        self.average_extinction = sum_extinction_per_group / count_per_group

    def regression(self, first=True):
        pool = mp.Pool(mp.cpu_count())

        if self.average_later_radio.isChecked():
            mp_inputs = [
                (
                    frame,
                    spot,
                    self.wavelengths,
                    self.extinction,
                    self.gp_radio.isChecked(),
                    self.poly_degree,
                )
                for frame in range(self.num_time_steps)
                for spot in range(self.num_spots)
            ]
            results = pool.starmap(individualRegression, mp_inputs)
            pool.close()
            pool.join()
            self.regressors = np.array(results).reshape(
                (self.num_time_steps, self.num_spots)
            )
        else:
            mp_inputs = [
                (
                    frame,
                    group,
                    self.wavelengths,
                    self.average_extinction,
                    self.gp_radio.isChecked(),
                    self.poly_degree,
                )
                for frame in range(self.num_time_steps)
                for group in range(self.num_groups)
            ]
            results = pool.starmap(individualRegression, mp_inputs)
            pool.close()
            pool.join()
            self.regressors = np.array(results).reshape(
                (self.num_time_steps, self.num_groups)
            )

        self.updateMetric(checked=True, first=first)

        metric_x = self.metric[:, :, 0]

        self.time_series_x = np.arange(self.num_time_steps)

        self.time_series_y = np.array(metric_x).reshape(
            self.num_time_steps, self.num_groups
        )

    def updateMetric(self, checked=True, first=False):
        if not checked:
            return
        if self.max_radio.isChecked():
            self.metric = np.array(
                [regressor.max() for regressor in self.regressors.flatten()]
            )
        if self.centroid_radio.isChecked():
            self.metric = np.array(
                [regressor.full_centroid() for regressor in self.regressors.flatten()]
            )
        if self.centroid_left_bound_radio.isChecked():
            self.metric = np.array(
                [
                    regressor.centroid_left_bound()
                    for regressor in self.regressors.flatten()
                ]
            )
        if self.centroid_half_height_radio.isChecked():
            self.metric = np.array(
                [
                    regressor.half_height_centroid()
                    for regressor in self.regressors.flatten()
                ]
            )
        if self.centroid_left_bound_half_height_radio.isChecked():
            self.metric = np.array(
                [
                    regressor.left_bound_half_height_centroid()
                    for regressor in self.regressors.flatten()
                ]
            )
        if self.inflection_radio.isChecked():
            self.metric = np.array(
                [regressor.inflection() for regressor in self.regressors.flatten()]
            )
        # if self.cross_correlation_radio.isChecked():
        #     self.metric = np.array(
        #         [
        #             regressor.bounded_cross_correlation(self.regressors.flatten()[0])
        #             for regressor in self.regressors.flatten()
        #         ]
        #     )
        # Compute metric to display
        if self.average_first_radio.isChecked():
            self.metric = self.metric.reshape((self.num_time_steps, self.num_groups, 2))
        else:
            self.metric = self.metric.reshape((self.num_time_steps, self.num_spots, 2))
            self.new_metric = np.zeros((self.num_time_steps, self.num_groups, 2))
            for i in range(self.num_groups):
                label = self.group_labels[i]
                count = 0
                for labels, index in zip(
                    self.selected_spot_labels, self.selected_spot_indices
                ):
                    if label in labels:
                        self.new_metric[:, i, :] += self.metric[:, index, :]
                        count += 1
                self.new_metric[:, i, :] /= count
            self.metric = self.new_metric

        if not first:
            self.updateMaxima()
            self.time_series_y = np.array(self.metric[:, :, 0]).reshape(
                self.num_time_steps, self.num_groups
            )
            self.parent.time_series.updateCurveData()
            self.parent.statistics_view.updateMeans()

    def updateTimeIndices(self, time_indices, time_labels):
        self.time_indices = time_indices
        self.time_labels = time_labels
        for curve in self.curves:
            curve.setVisible(False)
        for time in self.time_indices:

            for spot_index in self.spot_indices:
                i = time * self.num_spots + spot_index
                self.curves[time][spot_index].setVisible(True)
        self.updateDataPoints()

    def updateDataPoints(self):
        x_vals = self.wavelengths[self.time_indices[:, None], :]
        x_vals = np.tile(x_vals, (1, 1, len(self.spot_indices)))
        x_vals = x_vals.flatten()

        # Select extinction values for the given time indices and spot indices
        y_vals = self.extinction[
            self.time_indices[:, None], :, self.spot_indices
        ]  # shape (len(time_indices), num_wavelengths, len(spot_indices))
        y_vals = y_vals.flatten()  # Flatten to get shape (num_values, )
        self.scatter_data_points.setData(x=x_vals, y=y_vals)

    def draw(self):
        self.plot_widget.clear()

        if self.average_time_checkbox.isChecked():
            if self.average_first_radio.isChecked():
                self.updateTimeRanges()
                self.time_average_curves = np.empty((2, self.num_groups), dtype=object)
                for group_index, group_label in enumerate(self.group_labels):
                    self.plot_average_over_time(group_index, group_index, group_label)
            else:
                self.updateTimeRanges()
                self.time_average_curves = np.empty(
                    (2, len(self.selected_spot_indices)), dtype=object
                )
                for group_index, group_label in enumerate(self.group_labels):
                    for spot_labels, spot_index in zip(
                        self.selected_spot_labels, self.selected_spot_indices
                    ):
                        if group_label in spot_labels:
                            self.plot_average_over_time(
                                spot_index, group_index, group_label
                            )
        else:
            if self.average_first_radio.isChecked():
                for time_enum, time in enumerate(self.time_indices):

                    for group_index, group_label in enumerate(self.group_labels):
                        # Plot data points, polynomials, and store wavelength for maximum for every curve

                        self.plot(
                            time, time_enum, group_index, group_index, group_label
                        )
            else:
                for time_enum, time in enumerate(self.time_indices):
                    for group_index, group_label in enumerate(self.group_labels):
                        for spot_labels, spot_index in zip(
                            self.selected_spot_labels, self.selected_spot_indices
                        ):
                            if group_label in spot_labels:
                                # Plot data points, polynomials, and store wavelength for maximum for every curve
                                self.plot(
                                    time,
                                    time_enum,
                                    spot_index,
                                    group_index,
                                    group_label,
                                )

        # Plot Maxima
        self.scatter_maxima = pg.ScatterPlotItem()
        self.plot_widget.addItem(self.scatter_maxima)
        self.updateMaxima()
        if not self.maxima_checkbox.isChecked():
            self.scatter_maxima.setVisible(False)

        # Plot data points
        self.scatter_data_points = pg.ScatterPlotItem()
        self.plot_widget.addItem(self.scatter_data_points)
        x_vals = (
            np.tile(self.wavelengths, (self.num_spots, 1, 1))
            .transpose(1, 2, 0)
            .flatten()
        )
        y_vals = self.extinction.flatten()
        self.scatter_data_points.setData(x=x_vals, y=y_vals)
        if not self.raw_data_checkbox.isChecked():
            self.scatter_data_points.setVisible(False)

    def toggle_raw_data(self):
        if self.raw_data_checkbox.isChecked():
            self.scatter_data_points.setVisible(True)
        else:
            self.scatter_data_points.setVisible(False)

    def toggle_maxima(self):
        if self.maxima_checkbox.isChecked():
            self.scatter_maxima.setVisible(True)
        else:
            self.scatter_maxima.setVisible(False)

    def updateMaxima(self):
        color = []
        colors = []
        for label in self.group_labels:
            color.append(QColor(*color_palette[label]))
        for i in range(self.num_time_steps):
            colors.extend(color)

        self.scatter_maxima.setData(
            x=self.metric[:, :, 0].flatten(),
            y=self.metric[:, :, 1].flatten(),
            brush=colors,
        )

    def updateCurvesData(self):
        if self.average_first_radio.isChecked():
            for group_index in range(self.num_groups):
                if self.average_time_checkbox.isChecked():
                    self.updateAverageOverTime(group_index)
                else:
                    for time_enum, time in enumerate(self.time_indices):
                        # Plot data points, polynomials, and store wavelength for maximum for every curve
                        self.updateCurveData(time, group_index)
        else:
            for spot_index in range(len(self.selected_spot_indices)):
                if self.average_time_checkbox.isChecked():
                    self.updateAverageOverTime(spot_index)
                else:
                    for time_enum, time in enumerate(self.time_indices):
                        self.updateCurveData(time, spot_index)

        self.updateMaxima()

    def get_time_series(self):
        return self.time_series_x, self.time_series_y

    def updateTimeRanges(self):
        range1 = self.parent.time_series.time_region1.getRegion()
        # Calculate indices of time_series_x that are within range1
        self.time_range1_indices = np.where(
            (self.time_series_x >= range1[0]) & (self.time_series_x <= range1[1])
        )[0]
        range2 = self.parent.time_series.time_region2.getRegion()
        self.time_range2_indices = np.where(
            (self.time_series_x >= range2[0]) & (self.time_series_x <= range2[1])
        )[0]

    def get_statistics(self):
        # Calculate mean and std for range1
        range1_means = np.mean(self.time_series_y[self.time_range1_indices], axis=0)
        range1_std = np.std(self.time_series_y[self.time_range1_indices], axis=0)

        # Calculate mean and std for range2
        range2_means = np.mean(self.time_series_y[self.time_range2_indices], axis=0)
        range2_std = np.std(self.time_series_y[self.time_range2_indices], axis=0)

        # Calculate overall mean and std
        mean = np.mean(self.time_series_y, axis=0)
        std = np.std(self.time_series_y, axis=0)

        # Stack all means and stds
        means = np.vstack((mean, range1_means, range2_means))
        stds = np.vstack((std, range1_std, range2_std))
        return means, stds

    def get_groups(self):
        return self.group_labels

    def get_spot_count(self):
        return self.num_spots

    def export_extinction_to_csv(self, filename):
        # Sort the wavelengths in ascending order
        sorted_wavelengths = np.sort(self.wavelengths[0])

        # Flatten the data
        flattened_data = []
        for t in range(self.num_time_steps):
            for s in range(self.num_spots):
                # Sort the extinction values according to the sorted wavelengths
                sort_indices = np.argsort(self.wavelengths[t])
                sorted_extinction = self.extinction[t, sort_indices, s]
                row = [t, s] + sorted_extinction.tolist()
                flattened_data.append(row)

        # Write to CSV
        with open(filename, "w", newline="") as csvfile:
            csvwriter = csv.writer(csvfile)
            # Write header
            header = ["Timestep", "Spot"] + [
                f"{wavelength:.2f}" for wavelength in sorted_wavelengths
            ]
            csvwriter.writerow(header)
            # Write data
            csvwriter.writerows(flattened_data)

    def export(self):
        proposed_filename = os.path.join(os.curdir, "extinction_values.csv")
        filename, _ = QFileDialog.getSaveFileName(
            self.widget,
            "Export Extinction Values",
            proposed_filename,
            "CSV Files (*.csv)",
        )
        if filename:
            self.export_extinction_to_csv(filename)
