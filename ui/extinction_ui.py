from PySide6.QtWidgets import (
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
    QRadioButton,
    QGroupBox,
    QFormLayout,
    QCheckBox,
    QFileDialog,
    QMessageBox,
    QLabel,
    QSpinBox,
    QSizePolicy,
    QDoubleSpinBox,
    QPushButton,
)
from PySide6.QtGui import QColor, QPen
from PySide6.QtCore import Qt
import pyqtgraph as pg
import numpy as np
import csv
import os
import time

from enum import Enum

from line_fitting.line_fitting import No_Reg, Polynomial, GPRegression
from misc.colors import color_palette
from misc.profiling import ProfileContext
import multiprocessing as mp
from multiprocessing.pool import ThreadPool

from enum import Enum
import gc


class RegressorType(Enum):
    NO_REG = "No regression"
    POLY = "Polynomial"
    GP = "GP"


# TODO do something smarter here
max_num_time_ranges = 20


def individualRegression(
    f, i, wavelengths, extinction, regressor_type, poly_degree, min, max
):
    x = wavelengths[f]
    y = extinction[f, :, i]

    # Create a boolean mask for wavelengths within the specified range
    mask = (x >= min) & (x <= max)

    # Apply the mask to get the data for fitting
    x = x[mask]
    y = y[mask]

    if regressor_type == RegressorType.NO_REG:
        regressor = No_Reg(x, y)
    elif regressor_type == RegressorType.GP:
        regressor = GPRegression(x, y)
    elif regressor_type == RegressorType.POLY:
        regressor = Polynomial(x, y, poly_degree)
    else:
        raise ValueError(f"Unknown regressor type: {regressor_type}")

    regressor.fit()

    return regressor


class ExtinctionUi:
    def __init__(self, parent=None):
        self.parent = parent
        self.widget = QWidget()
        self.layout = QHBoxLayout()
        self.plot_widget = pg.PlotWidget()
        # self.plot_widget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.layout.addWidget(self.plot_widget)
        self.button_layout = QVBoxLayout()
        # self.selection_layout = QVBoxLayout()
        # self.layout.addLayout(self.selection_layout)
        self.extinction_display_options()
        self.group_averaging_options()
        # self.selection_layout.addStretch()
        self.plot_widget.setBackground("w")
        self.plot_widget.setTitle("Extinction", color="black")
        self.plot_widget.setLabel("left", "Extinction (a.u.)")
        self.plot_widget.setLabel("bottom", "Wavelength", units="nm")
        self.spot_labels = None
        self.time_indices = None
        self.time_labels = None
        self.regressors = None
        self.averaged_regressors = None
        self.num_time_steps = None
        self.num_spots = None
        self.curves = None
        self.num_regression_points = 40
        self.time_range_indices = []
        self.metric = None
        self.averaged_metric = None
        self.time_average_curves = None
        # TODO do I need to set this to None, after selected points changed?
        self.individual_metric = None
        self.min_wavelength_data = 450
        self.max_wavelength_data = 700
        self.default_selected_min_wavelength = self.min_wavelength_data
        self.default_selected_max_wavelength = self.max_wavelength_data

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

        if self.averaged_regressors is not None:
            for row in range(self.averaged_regressors.shape[0]):
                for col in range(self.averaged_regressors.shape[1]):
                    regressor = self.averaged_regressors[row, col]
                    if regressor is not None:
                        regressor.cleanup()  # Call the cleanup method of each regressor

        self.averaged_regressors = None

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
        self.spot_labels = None
        self.time_indices = None
        self.time_labels = None

    def computeEverything(self):
        self.individual_metric = None
        self.regressors = None
        self.averaged_regressors = None
        gc.collect()
        self.computeExtinction()
        self.setAllIndices()
        # TODO: Its not possible to set more groups than twice the number of spots.Probably not necessary but this should at least be checked properly
        if self.curves is None:
            self.curves = np.empty(
                (self.num_time_steps, self.num_spots * 2), dtype=object
            )
        self.generate_group_lists()
        if self.average_first_radio.isChecked():
            self.averageGroups()
        self.regression(
            self.average_first_radio.isChecked(),
            self.min_wavelength_data,
            self.max_wavelength_data,
        )

    def extinction_display_options(self):
        self.extinction_display_group_box = QGroupBox("Extinction Display")
        group_layout = QVBoxLayout()
        self.extinction_display_group_box.setLayout(group_layout)
        # self.selection_layout.addWidget(group_box)

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
        self.group_averaging_group_box = QGroupBox("Group Averaging")
        group_layout = QVBoxLayout()
        self.group_averaging_group_box.setLayout(group_layout)
        # self.selection_layout.addWidget(group_box)

        self.average_first_radio = QRadioButton("Average extinction data")
        self.average_later_radio = QRadioButton("Average calculated metric")
        self.average_first_radio.setChecked(True)

        group_layout.addWidget(self.average_first_radio)
        group_layout.addWidget(self.average_later_radio)

        self.average_first_radio.toggled.connect(self.updateAverageGroups)

    def updateAverageGroups(self):
        if self.average_first_radio.isChecked():
            if self.averaged_regressors is None:
                self.averageGroups()
                self.regression(
                    self.average_first_radio.isChecked(),
                    self.min_wavelength_spin.value(),
                    self.max_wavelength_spin.value(),
                    first=False,
                )
        else:
            if self.regressors is None:
                self.regression(
                    self.average_first_radio.isChecked(),
                    self.min_wavelength_spin.value(),
                    self.max_wavelength_spin.value(),
                    first=False,
                )
        self.draw()
        self.parent.time_series.updateCurveData()

    def regressor_selection_ui(self):
        self.reg_selection_group_box = QGroupBox("Select Regressor & Wavelength Range")
        group_layout = QVBoxLayout()

        # --- Regressor Selection ---
        self.no_reg_radio = QRadioButton(RegressorType.NO_REG.value)
        self.poly_radio = QRadioButton(RegressorType.POLY.value)
        self.gp_radio = QRadioButton(RegressorType.GP.value)

        self.no_reg_radio.setChecked(True)  # Default selection

        self.regressor_map = {
            self.no_reg_radio: RegressorType.NO_REG,
            self.poly_radio: RegressorType.POLY,
            self.gp_radio: RegressorType.GP,
        }

        self.poly_degree_spin = QSpinBox()
        self.poly_degree_spin.setValue(3)
        self.poly_degree_spin.setRange(1, 10)
        self.poly_degree_spin.setEnabled(False)
        self.poly_degree_spin.setStyleSheet("QSpinBox { color: gray; }")
        self.poly_degree_spin.setToolTip("Set polynomial degree.")
        # self.poly_degree_spin.valueChanged.connect(self.updateRegressor) # DISCONNECTED

        group_layout.addWidget(self.no_reg_radio)
        poly_layout = QHBoxLayout()
        poly_layout.addWidget(self.poly_radio)
        poly_layout.addWidget(self.poly_degree_spin)
        poly_layout.addStretch()
        group_layout.addLayout(poly_layout)
        group_layout.addWidget(self.gp_radio)

        # --- Wavelength Range Selection ---
        wavelength_group_box = QGroupBox("Wavelength Range (nm)")
        wavelength_layout = QHBoxLayout()

        min_label = QLabel("Min:")
        self.min_wavelength_spin = QSpinBox()
        self.min_wavelength_spin.setRange(
            self.min_wavelength_data, self.max_wavelength_data
        )
        self.min_wavelength_spin.setValue(self.default_selected_min_wavelength)
        self.min_wavelength_spin.setSuffix(" nm")
        self.min_wavelength_spin.setToolTip("Minimum wavelength for fitting.")
        # self.min_wavelength_spin.valueChanged.connect(self.updateWavelengthRange) # DISCONNECTED

        max_label = QLabel("Max:")
        self.max_wavelength_spin = QSpinBox()
        self.max_wavelength_spin.setRange(
            self.min_wavelength_data, self.max_wavelength_data
        )
        self.max_wavelength_spin.setValue(self.default_selected_max_wavelength)
        self.max_wavelength_spin.setSuffix(" nm")
        self.max_wavelength_spin.setToolTip("Maximum wavelength for fitting.")
        # self.max_wavelength_spin.valueChanged.connect(self.updateWavelengthRange) # DISCONNECTED

        wavelength_layout.addWidget(min_label)
        wavelength_layout.addWidget(self.min_wavelength_spin)
        wavelength_layout.addSpacing(20)
        wavelength_layout.addWidget(max_label)
        wavelength_layout.addWidget(self.max_wavelength_spin)
        wavelength_layout.addStretch()

        wavelength_group_box.setLayout(wavelength_layout)
        group_layout.addWidget(wavelength_group_box)

        # --- Update Button ---
        self.update_regression_button = QPushButton("Update Regression")
        self.update_regression_button.setToolTip(
            "Apply selected regressor and wavelength range settings."
        )
        # Connect the button to the new unified update method
        self.update_regression_button.clicked.connect(self.updateRegressor)

        # Add a spacer to push the button to the bottom
        group_layout.addStretch()
        group_layout.addWidget(self.update_regression_button)

        self.reg_selection_group_box.setLayout(group_layout)

    def metric_selection_ui(self):
        self.metric_selection_group_box = QGroupBox("Select Metric")
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

        self.centroid_radio.setChecked(True)
        # self.cross_correlation_radio.setChecked(True)

        group_layout.addWidget(self.max_radio)
        group_layout.addWidget(self.centroid_radio)
        # group_layout.addWidget(self.centroid_left_bound_radio)
        # group_layout.addWidget(self.centroid_half_height_radio)
        # group_layout.addWidget(self.centroid_left_bound_half_height_radio)
        group_layout.addWidget(self.inflection_radio)
        # group_layout.addWidget(self.cross_correlation_radio)

        self.metric_selection_group_box.setLayout(group_layout)

        # self.button_layout.addWidget(group_box)

        self.max_radio.toggled[bool].connect(
            lambda checked: self.updateMetric(
                self.average_first_radio.isChecked(), checked, False
            )
        )
        self.centroid_radio.toggled[bool].connect(
            lambda checked: self.updateMetric(
                self.average_first_radio.isChecked(), checked, False
            )
        )

        self.centroid_left_bound_radio.toggled[bool].connect(
            lambda checked: self.updateMetric(
                self.average_first_radio.isChecked(), checked, False
            )
        )
        self.centroid_half_height_radio.toggled[bool].connect(
            lambda checked: self.updateMetric(
                self.average_first_radio.isChecked(), checked, False
            )
        )
        self.centroid_left_bound_half_height_radio.toggled[bool].connect(
            lambda checked: self.updateMetric(
                self.average_first_radio.isChecked(), checked, False
            )
        )
        self.inflection_radio.toggled[bool].connect(
            lambda checked: self.updateMetric(
                self.average_first_radio.isChecked(), checked, False
            )
        )
        # self.cross_correlation_radio.toggled[bool].connect(self.updateMetric)

    def updateRegressor(self):
        if self.poly_radio.isChecked():
            self.poly_degree_spin.setEnabled(True)
            self.poly_degree_spin.setStyleSheet("")
        else:
            self.poly_degree_spin.setEnabled(False)
            self.poly_degree_spin.setStyleSheet("QSpinBox { color: gray; }")
        self.averaged_regressors = None
        self.regressors = None
        self.individual_metric = None
        gc.collect()
        min = self.min_wavelength_spin.value()
        max = self.max_wavelength_spin.value()
        self.regression(self.average_first_radio.isChecked(), min, max, first=False)
        self.updateCurvesData()
        self.parent.time_series.updateCurveData()
        if self.average_time_checkbox.isChecked():
            self.updateAverageGroups()

    def plot(self, time, time_label_index, spot_index, group_index, spot_label):
        if self.average_first_radio.isChecked():
            regressor = self.averaged_regressors[time][group_index]
        else:
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
        if self.no_reg_radio.isChecked():
            regressor = No_Reg(x_mean, y_mean)
        elif self.poly_radio.isChecked():
            regressor = Polynomial(x_mean, y_mean, self.poly_degree_spin.value())
        elif self.gp_radio.isChecked():
            regressor = GPRegression(x_mean, y_mean)

        # Fit regressor and generate plot data
        regressor.fit()
        return regressor.generateValues(self.num_regression_points)

    def plot_average_over_time(self, spot_index, group_index, spot_label):
        # Create and add plot line
        spot_color = color_palette[spot_label]

        for i, time_range in enumerate(self.time_range_indices):
            pen = pg.mkPen(color=QColor(*spot_color), width=3)
            x_plot, y_plot = self.compute_average_over_time(time_range, spot_index)
            line = pg.PlotDataItem(x_plot, y_plot, pen=pen, symbol=None)
            line.setZValue(-1 * i)
            self.plot_widget.addItem(line)
            self.time_average_curves[i, spot_index] = line

    def updateAverageOverTime(self, group_index):
        for i, time_range in enumerate(self.time_range_indices):
            x, y = self.compute_average_over_time(time_range, group_index)
            self.time_average_curves[i, group_index].setData(x, y)

    def toggle_time_average(self):
        self.draw()

    def updateCurveData(self, time, group_index):
        if self.average_first_radio.isChecked():
            regressor = self.averaged_regressors[time][group_index]
        else:
            regressor = self.regressors[time][group_index]
        x_values, y_values = regressor.generateValues(self.num_regression_points)
        self.curves[time][group_index].setData(x_values, y_values)

    def setAllIndices(self):
        self.time_indices = np.array(range(self.num_time_steps))
        self.time_labels = np.zeros_like((self.time_indices))
        self.spot_labels = np.zeros(self.extinction[0].shape[1])

    def computeExtinction(self):

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
                sum_extinction_per_group[:, :, group_index] += self.extinction[:, :, i]
                count_per_group[group_index] += 1

        # Compute the average extinction for each group
        self.average_extinction = sum_extinction_per_group / count_per_group

    def regression(self, average_first, min, max, first=True):

        # Select the appropriate extinction data and regressor storage
        extinction_data = self.average_extinction if average_first else self.extinction
        regressor_storage = "averaged_regressors" if average_first else "regressors"

        if self.no_reg_radio.isChecked():
            selected_regressor = RegressorType.NO_REG
        elif self.poly_radio.isChecked():
            selected_regressor = RegressorType.POLY
        elif self.gp_radio.isChecked():
            selected_regressor = RegressorType.GP

        pool = mp.Pool(mp.cpu_count())

        mp_inputs = [
            (
                frame,
                index,
                self.wavelengths,
                extinction_data,
                selected_regressor,
                self.poly_degree_spin.value(),
                min,
                max,
            )
            for frame in range(self.num_time_steps)
            for index in range(self.num_groups if average_first else self.num_spots)
        ]

        t0 = time.time()
        with ThreadPool(processes=4) as pool:
            results = pool.starmap(individualRegression, mp_inputs)
        print(f"Extinction multiprocess {time.time() - t0:.2f} seconds")
        pool.close()
        pool.join()

        setattr(
            self,
            regressor_storage,
            np.array(results).reshape(
                (
                    self.num_time_steps,
                    self.num_groups if average_first else self.num_spots,
                )
            ),
        )

        self.updateMetric(average_first, checked=True, first=first)

        metric = self.averaged_metric if average_first else self.metric
        self.time_series_x = np.arange(self.num_time_steps)
        self.time_series_y = np.array(metric[:, :, 0]).reshape(
            self.num_time_steps, self.num_groups
        )

    def updateMetric(self, average_first, checked=True, first=False):
        if self.no_reg_radio.isChecked():
            if not (self.max_radio.isChecked() or self.centroid_radio.isChecked()):
                QMessageBox.warning(
                    self.parent,
                    "Not Implemented",
                    "This metric is not implemented for No Regression. Using Centroid instead.",
                )
                self.centroid_radio.setChecked(True)

        if not checked:
            return

        regressors = self.averaged_regressors if average_first else self.regressors
        metric_storage = "averaged_metric" if average_first else "metric"

        if self.max_radio.isChecked():
            metric = [regressor.max() for regressor in regressors.flatten()]
        elif self.centroid_radio.isChecked():
            metric = [regressor.full_centroid() for regressor in regressors.flatten()]
        elif self.centroid_left_bound_radio.isChecked():
            metric = [
                regressor.centroid_left_bound() for regressor in regressors.flatten()
            ]
        elif self.centroid_half_height_radio.isChecked():
            metric = [
                regressor.half_height_centroid() for regressor in regressors.flatten()
            ]
        elif self.centroid_left_bound_half_height_radio.isChecked():
            metric = [
                regressor.left_bound_half_height_centroid()
                for regressor in regressors.flatten()
            ]
        elif self.inflection_radio.isChecked():
            metric = [regressor.inflection() for regressor in regressors.flatten()]
        else:
            metric = []

        metric = np.array(metric).reshape(
            (
                self.num_time_steps,
                self.num_groups if average_first else self.num_spots,
                2,
            )
        )

        setattr(self, metric_storage, metric)

        if not average_first:
            # Aggregate metrics for groups
            self.new_metric = np.zeros((self.num_time_steps, self.num_groups, 2))
            for i in range(self.num_groups):
                label = self.group_labels[i]
                count = 0
                for index, labels in enumerate(self.selected_spot_labels):
                    if label in labels:
                        self.new_metric[:, i, :] += self.metric[:, index, :]
                        count += 1
                self.new_metric[:, i, :] /= count
            self.individual_metric = self.metric
            self.metric = self.new_metric

        if not first:
            self.updateMaxima()
            metric = self.averaged_metric if average_first else self.metric
            self.time_series_y = np.array(metric[:, :, 0]).reshape(
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

            for spot_index in range(self.spot_labels.shape[0]):
                self.curves[time][spot_index].setVisible(True)
        self.updateDataPoints()

    def updateDataPoints(self):
        x_vals = self.wavelengths[self.time_indices[:, None], :]
        x_vals = np.tile(x_vals, (1, 1, self.num_spots))
        x_vals = x_vals.flatten()

        # Select extinction values for the given time indices and spot indices
        y_vals = self.extinction[
            self.time_indices[:, None], :, :
        ]  # shape (len(time_indices), num_wavelengths, len(spot_indices))
        y_vals = y_vals.flatten()  # Flatten to get shape (num_values, )
        self.scatter_data_points.setData(x=x_vals, y=y_vals)

    def draw(self):
        self.plot_widget.clear()

        if self.average_time_checkbox.isChecked():
            if self.average_first_radio.isChecked():
                self.updateTimeRanges()
                self.time_average_curves = np.empty(
                    (max_num_time_ranges, self.num_groups), dtype=object
                )
                for group_index, group_label in enumerate(self.group_labels):
                    self.plot_average_over_time(group_index, group_index, group_label)
            else:
                self.updateTimeRanges()
                self.time_average_curves = np.empty(
                    (max_num_time_ranges, self.num_spots), dtype=object
                )
                for group_index, group_label in enumerate(self.group_labels):
                    for spot_index, spot_labels in enumerate(self.selected_spot_labels):
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
                        for spot_index, spot_labels in enumerate(
                            self.selected_spot_labels
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
        self.updateCurveDashes()

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
        labels = [inner_list[0] for inner_list in self.selected_spot_labels]
        colors = []
        for label in labels:
            color = QColor(*color_palette[label])
            colors.append(color)
        large_color_array = np.tile(colors, np.prod(self.wavelengths.shape))
        self.scatter_data_points.setData(x=x_vals, y=y_vals, brush=large_color_array)
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

        if self.average_first_radio.isChecked():
            self.scatter_maxima.setData(
                x=self.averaged_metric[:, :, 0].flatten(),
                y=self.averaged_metric[:, :, 1].flatten(),
                brush=colors,
            )
        else:
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
            for spot_index in range(self.num_spots):
                if self.average_time_checkbox.isChecked():
                    self.updateAverageOverTime(spot_index)
                else:
                    for time_enum, time in enumerate(self.time_indices):
                        self.updateCurveData(time, spot_index)

        self.updateMaxima()

    def updateCurveDashes(self):
        dash_lengths = [
            0.01,
            1.5,
            3.2,
            5.2,
            7.5,
            10.1,
            13.1,
            16.6,
            20.6,
            25.2,
            30.5,
            36.6,
            43.6,
        ]
        if self.average_first_radio.isChecked():
            for group_index in range(self.num_groups):
                if self.average_time_checkbox.isChecked():
                    for i, time_range in enumerate(self.time_range_indices):
                        original_pen = self.time_average_curves[i][group_index].opts[
                            "pen"
                        ]
                        pen = QPen(original_pen)
                        pen.setWidth(4)
                        pen.setCosmetic(True)
                        pen.setCapStyle(Qt.RoundCap)
                        if i != 0:
                            dash_length = dash_lengths[(i - 1) % len(dash_lengths)] * 2
                            gap_length = dash_length + 4 * 2
                            pen.setDashPattern([dash_length, gap_length])
                        self.time_average_curves[i][group_index].setPen(pen)
        else:
            for spot_index in range(self.num_spots):
                if self.average_time_checkbox.isChecked():
                    for i, time_range in enumerate(self.time_range_indices):
                        if i != 0:
                            original_pen = self.time_average_curves[i][spot_index].opts[
                                "pen"
                            ]
                            pen = QPen(original_pen)
                            dash_length = dash_lengths[(i - 1) % len(dash_lengths)]
                            gap_length = dash_length + 4
                            pen.setDashPattern([dash_length, gap_length])
                            self.time_average_curves[i][spot_index].setPen(pen)

    def get_time_series(self):
        return self.time_series_x, self.time_series_y

    def updateTimeRanges(self):
        self.time_range_indices = [None] * len(self.parent.time_series.time_ranges)
        for i, range in enumerate(self.parent.time_series.time_ranges):
            region = range.getRegion()
            self.time_range_indices[i] = np.where(
                (self.time_series_x >= region[0]) & (self.time_series_x <= region[1])
            )[0]

    def get_data_for_results_display(self):
        # We can't use get_statistics directly since we need the statistics for individual spots
        if self.individual_metric is None:
            self.regression(average_first=False)

        diffs = []
        diff_sems = []
        reference_mean = None
        reference_sem = None
        for i, range in enumerate(self.time_range_indices):
            values = self.individual_metric[range, :, 0]
            mean = np.mean(values, axis=0)
            sem = np.std(values, axis=0) / np.sqrt(len(range))
            if i == 0:
                reference_mean = mean
                reference_sem = sem
            else:
                diff = mean - reference_mean
                diff_sem = np.sqrt(sem**2 + reference_sem**2)
                diffs.append(diff)
                diff_sems.append(diff_sem)
        return np.array(diffs), np.array(diff_sems)

    def get_statistics(self):

        means = []
        sems = []
        for range in self.time_range_indices:
            mean = np.mean(self.time_series_y[range], axis=0)
            sem = np.std(self.time_series_y[range], axis=0) / np.sqrt(len(range))
            means.append(mean)
            sems.append(sem)
        return np.array(means), np.array(sems)

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
