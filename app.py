import sys
import random

from PySide6.QtWidgets import (
    QFileDialog,
    QApplication,
    QMainWindow,
    QStackedLayout,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QWidget,
    QPushButton,
    QRadioButton,
    QGroupBox,
)
from PySide6.QtGui import (
    QAction,
)

from ui.spot_detection_ui import SpotDetectionUi
from ui.spot_selection_ui import SpotSelectionUi
from ui.extinction_ui import ExtinctionUi
from ui.time_series_ui import TimeSeries
from ui.statistics_view import StatisticsView
from misc.profiling import ProfileContext
import gc


class MainWindow(QMainWindow):
    def __init__(self, path=None):
        super(MainWindow, self).__init__()
        self.setWindowTitle("MIKA")

        # Save the current path
        # self.path = path if path else "/home/asterzik/projects/mika/data/cropped"
        self.path = (
            path
            if path
            else "/media/sd/mika/data/Calibration_water_5xSSC_6LEDs/cropped/20_timesteps"
            # else "/media/sd/mika/data/Calibration_water_5xSSC_6LEDs/Images"
        )

        # Initialize variables for UI components (to be deleted later)
        self.spot_ui = None
        self.spot_selection_ui = None
        self.extinction_ui = None
        self.time_series = None

        self.initUI()

    def initUI(self):
        """Initialize the UI elements."""
        central_widget = QWidget(self)  # Set parent to the main window
        self.setCentralWidget(central_widget)

        pagelayout = QHBoxLayout()
        main_content_layout = QVBoxLayout()
        pipeline_button_layout = QHBoxLayout()
        self.pipeline_stack_layout = QStackedLayout()
        sidebar_layout = QVBoxLayout()

        pagelayout.addLayout(main_content_layout)
        # pagelayout.addLayout(sidebar_layout)
        main_content_layout.addLayout(pipeline_button_layout)
        main_content_layout.addLayout(self.pipeline_stack_layout)

        central_widget.setLayout(pagelayout)

        # Initialize UI elements with the current path
        self.spot_ui = SpotDetectionUi(self.path, self)
        self.spot_selection_ui = SpotSelectionUi(self.path, self)
        self.extinction_ui = ExtinctionUi(self)
        self.time_series = TimeSeries(self)
        # Simulated statistics data (you would replace this with actual calculations)
        self.statistics_view = StatisticsView(self)

        # Create an analysis widget layout
        self.analysis_widget = QWidget(self)
        self.analysis_layout = QHBoxLayout(self.analysis_widget)
        self.extinction_layout = QVBoxLayout()
        self.time_layout = QVBoxLayout()

        self.analysis_layout.addLayout(self.extinction_layout)
        self.analysis_layout.addLayout(self.time_layout)
        self.analysis_layout.addLayout(sidebar_layout)

        # Add widgets to layouts
        self.extinction_layout.addWidget(self.extinction_ui.widget, stretch=2)
        self.extinction_layout.addWidget(self.spot_selection_ui.widget, stretch=1)
        self.time_layout.addWidget(self.time_series.widget, stretch=3)
        self.time_layout.addWidget(self.statistics_view, stretch=1)

        self.pipeline_stack_layout.addWidget(self.spot_ui.widget)
        self.pipeline_stack_layout.addWidget(self.analysis_widget)

        # Add pipeline buttons
        spot_detection_layer_button = QPushButton("Spot Detection", self)
        spot_detection_layer_button.clicked.connect(
            lambda: self.pipeline_stack_layout.setCurrentIndex(0)
        )

        analysis_layer_button = QPushButton("Analysis", self)
        analysis_layer_button.clicked.connect(self.analysis_layer_button_clicked)

        pipeline_button_layout.addWidget(spot_detection_layer_button)
        pipeline_button_layout.addWidget(analysis_layer_button)
        # Denoising buttons
        mean_layout = QFormLayout()
        mean_group_box = QGroupBox("Mean computation", self)
        mean_group_box.setLayout(mean_layout)

        # Create a list of radio buttons with their corresponding labels
        self.mean_buttons = {
            "Mean": QRadioButton("Mean", self),
            "Median": QRadioButton("Median", self),
            "Mode": QRadioButton("Mode", self),
            "Trimmed Mean": QRadioButton("Trimmed Mean", self),
            "Huber Mean": QRadioButton("Huber Mean", self),
        }

        # Set the default checked button
        self.mean_buttons["Mean"].setChecked(True)
        self.mean_method = "Mean"

        # Add buttons to the layout and connect their toggled signals to a single method
        for method, button in self.mean_buttons.items():
            mean_layout.addWidget(button)
            button.toggled.connect(self.set_mean_method)

        # Add the denoising group box to the sidebar layout
        sidebar_layout.addWidget(mean_group_box)

        denoising_layout = QFormLayout()
        denoising_group_box = QGroupBox("Denoising", self)
        denoising_group_box.setLayout(denoising_layout)

        # Create a list of radio buttons with their corresponding labels
        self.denoising_buttons = {
            "None": QRadioButton("None", self),
            "HyRes": QRadioButton("HyRes", self),
            "HyMinor": QRadioButton("HyMinor", self),
            "FastHyDe": QRadioButton("FastHyDe", self),
            "L1HyMixDe": QRadioButton("L1HyMixDe", self),
            "WSRRR": QRadioButton("WSRRR"),
            "OTVCA": QRadioButton("OTVCA"),
            "FORPDN": QRadioButton("FORPDN"),
        }

        # Set the default checked button
        self.denoising_buttons["None"].setChecked(True)
        self.denoising_method = "None"

        # Add buttons to the layout and connect their toggled signals to a single method
        for method, button in self.denoising_buttons.items():
            denoising_layout.addWidget(button)
            button.toggled.connect(self.set_denoising_method)

        # TODO: Really check out denoising options and make sure they work if they are to be selected in the framework
        # # Add the denoising group box to the sidebar layout
        # sidebar_layout.addWidget(denoising_group_box)

        sidebar_layout.addLayout(self.extinction_ui.button_layout)

        # Add export section to the sidebar layout
        export_layout = QVBoxLayout()
        export_group_box = QGroupBox("Export Data", self)
        export_group_box.setLayout(export_layout)

        export_extinction_button = QPushButton("Export Extinction", self)
        export_timeseries_button = QPushButton("Export Time Series", self)
        export_extinction_button.clicked.connect(self.extinction_ui.export)
        export_timeseries_button.clicked.connect(self.time_series.export)

        export_layout.addWidget(export_extinction_button)
        export_layout.addWidget(export_timeseries_button)
        sidebar_layout.addWidget(export_group_box)

        # Open Action
        self.open_action = QAction("&Open", self)
        self.open_action.triggered.connect(self.openFolder)
        self.createMenu()

    def analysis_layer_button_clicked(self):
        # with ProfileContext("start_analysis.prof"):
        if self.spot_ui.circles.selected_spots == []:
            self.spot_ui.circles.select_all_spots(self.spot_ui.interactive_image)
        self.pipeline_stack_layout.setCurrentIndex(1)
        self.extinction_ui.computeEverything()
        self.time_series.draw()
        self.extinction_ui.draw()
        means, stds = self.extinction_ui.get_statistics()
        group_labels = self.extinction_ui.get_groups()
        self.statistics_view.init_groups(means, stds, group_labels)
        self.spot_selection_ui.update_image()

    def set_mean_method(self):
        # Update the denoising method based on the checked button
        self.mean_method = next(
            method for method, button in self.mean_buttons.items() if button.isChecked()
        )

        # Call the compute extinction method
        self.spot_ui.circles.compute_extinction()

        if self.pipeline_stack_layout.currentIndex() == 1:
            self.extinction_ui.computeEverything()
            self.extinction_ui.updateDataPoints()
            self.extinction_ui.updateCurvesData()
            self.time_series.updateCurveData()

    def set_denoising_method(self):
        # Update the denoising method based on the checked button
        denoising_method = next(
            method
            for method, button in self.denoising_buttons.items()
            if button.isChecked()
        )

        # Call the compute extinction method
        self.spot_ui.circles.load_images(self.path, denoising_method)
        self.spot_selection_ui.redrawImage()
        self.spot_ui.interactive_image.displayRepresentativeImage()
        self.spot_ui.circles.compute_extinction()

        if self.pipeline_stack_layout.currentIndex() == 1:
            self.extinction_ui.computeEverything(),
            self.extinction_ui.updateCurvesData(),
            self.extinction_ui.updateDataPoints(),
            self.time_series.updateCurveData(),

    def keyPressEvent(self, event):
        # Pass on the key press event to the spot_ui
        self.spot_ui.keyPressEvent(event)

    def createMenu(self):
        menu = self.menuBar()
        file_menu = menu.addMenu("&File")

        # Add Open action to the File menu
        file_menu.addAction(self.open_action)

    def openFolder(self):
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.Directory)
        if file_dialog.exec():
            self.path = file_dialog.selectedFiles()[0]
            self.resetUI()  # Call method to reset the UI

    def resetUI(self):
        """Completely destroy current UI and reinitialize from scratch."""

        # 1. Stop any timers, threads, or background tasks
        if hasattr(self, "background_thread") and self.background_thread:
            self.background_thread.terminate()
            self.background_thread.wait()
            self.background_thread = None

        if hasattr(self, "update_timer") and self.update_timer:
            self.update_timer.stop()
            self.update_timer = None

        # 2. Cleanup UI components
        for attr in ["spot_ui", "spot_selection_ui", "extinction_ui", "time_series"]:
            ui_element = getattr(self, attr, None)
            if ui_element:
                ui_element.cleanup()
                setattr(self, attr, None)

        # 3. Remove the central widget
        central_widget = self.centralWidget()
        if central_widget:
            central_widget.deleteLater()
            self.setCentralWidget(None)  # Ensures full removal

        # 4. Clear all menus and actions
        menu_bar = self.menuBar()
        if menu_bar:
            menu_bar.clear()
            for action in menu_bar.actions():
                menu_bar.removeAction(action)

        # 5. Force garbage collection to clean up any lingering objects
        gc.collect()

        # 6. Reinitialize the UI from scratch
        self.initUI()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
