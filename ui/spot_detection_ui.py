import numpy as np
from PySide6.QtWidgets import (
    QLabel,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QDoubleSpinBox,
    QGroupBox,
    QFormLayout,
    QToolBar,
    QSpacerItem,
    QSizePolicy,
    QButtonGroup,
    QRubberBand,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsPixmapItem,
    QToolBox,
)
from PySide6.QtGui import (
    QAction,
    QMouseEvent,
    QKeyEvent,
    QActionGroup,
    QPainter,
    QPen,
    QColor,
)
from PySide6.QtCore import Qt, QPoint, QRect, QSize

from image_processing.circle_detection import Circles
from misc.colors import color_palette, time_color_palette
import pyqtgraph as pg


class InteractiveImage(QGraphicsView):
    def __init__(self, ui, circles):
        super().__init__(ui.parent)

        # Set up the scene
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        # Create a pixmap item to hold the image
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)

        # Enable scroll bars and set the background color
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # Other member variables
        self.ui = ui
        self.circles = circles
        self.rubber_band = QRubberBand(QRubberBand.Rectangle, self)
        self.origin = QPoint()
        self.current_rect = QRect()
        self.is_dragging = False
        self.click_threshold = 5  # Minimum distance to differentiate click and drag

        # Set the size policy to expanding
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def setPixmap(self, pixmap):
        self.pixmap_item.setPixmap(pixmap)
        self.adjustImage()  # Adjust the image to fit the view

    def displayRepresentativeImage(self):
        pixmap = self.ui.circles.get_representative_image()
        self.setPixmap(pixmap)

    def adjustImage(self):
        # Set the scene rect to match the pixmap
        self.setSceneRect(self.pixmap_item.pixmap().rect())

        # Scale the view to fit the pixmap while maintaining aspect ratio
        self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)

    def keyPressEvent(self, event):
        self.ui.keyPressEvent(event)

    def mousePressEvent(self, event: QMouseEvent):
        x, y = self.transform_coordinates(event.pos().x(), event.pos().y())
        self.ui.mousePressInImage(x, y)
        if event.button() == Qt.LeftButton:
            self.origin = event.pos()
            self.is_dragging = False  # Reset dragging state
            self.rubber_band.setGeometry(
                QRect(self.origin, QSize())
            )  # Initialize the rubber band
            self.rubber_band.show()  # Show the rubber band

    def mouseMoveEvent(self, event: QMouseEvent):
        if event.buttons() & Qt.LeftButton:  # Check if the left mouse button is pressed
            # Check if the mouse moved more than the threshold
            if (
                not self.is_dragging
                and (event.pos() - self.origin).manhattanLength() > self.click_threshold
            ):
                self.is_dragging = True

            if self.is_dragging:
                self.current_rect = QRect(self.origin, event.pos()).normalized()
                self.rubber_band.setGeometry(self.current_rect)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.rubber_band.hide()  # Hide the rubber band
            if self.is_dragging:
                # Handle rectangle selection (dragged mouse)

                self.current_rect = QRect(self.origin, event.pos()).normalized()
                p1x, p1y = self.transform_coordinates(self.origin.x(), self.origin.y())

                p2x, p2y = self.transform_coordinates(event.pos().x(), event.pos().y())

                self.circles.select_spots(
                    QRect(QPoint(p1x, p1y), QPoint(p2x, p2y)), self.ui.current_group
                )
                # self.ui.updateSpotIndices()
                self.displayRepresentativeImage()
                self.circles.highlight_selected(self)
            else:
                # Handle circle selection (clicked mouse)
                x, y = self.origin.x(), self.origin.y()
                self.select_circle(x, y, self.ui.current_group)
                # self.ui.updateSpotIndices()

            self.is_dragging = False  # Reset dragging state

    def draw_measurement(self, start_point, end_point, measured_dist):
        pixmap = self.pixmap_item.pixmap()
        painter = QPainter(pixmap)
        pen = QPen(QColor("#A4031F"))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawLine(start_point, end_point)
        painter.drawText(end_point, f"{round(measured_dist)} pixels")
        painter.end()  # Finish painting
        self.setPixmap(pixmap)

    def transform_coordinates(self, x, y):
        # Convert the mouse coordinates from view coordinates to scene coordinates
        scene_point = self.mapToScene(x, y)  # Directly map to scene coordinates

        # Get the pixmap's original size
        pixmap = self.pixmap_item.pixmap()
        pixmap_width = pixmap.width()
        pixmap_height = pixmap.height()

        # Get the bounding rect of the pixmap item in the scene
        pixmap_item_rect = self.pixmap_item.boundingRect()

        # Calculate scale factors based on the bounding rect and pixmap dimensions
        scale_x = pixmap_width / pixmap_item_rect.width()
        scale_y = pixmap_height / pixmap_item_rect.height()

        # Map the scene coordinates back to pixmap coordinates
        pixmap_x = (scene_point.x() - pixmap_item_rect.x()) * scale_x
        pixmap_y = (scene_point.y() - pixmap_item_rect.y()) * scale_y

        return pixmap_x, pixmap_y

    def select_circle(self, x, y, group=0):
        # self.setPixmap(self.circles.draw_circles())
        self.circles.select_spot(x, y, group)
        # self.highlight_circle(group)

    def deselect_spots(self):
        self.circles.deselect_spots()

    def select_all_spots(self):
        self.circles.select_all_spots(self, self.ui.current_group)

    def highlight_circle_by_coordinates(self, x, y, r, group=None, shift=None):
        width = 2
        pixmap = self.pixmap_item.pixmap()
        if shift is None:
            shift = [0, 0]
        if group is None:
            rgb_color = (0, 0, 0)
            pen = QPen(QColor(*rgb_color))
            pen.setWidth(width)

            painter = QPainter(pixmap)
            painter.setPen(pen)

            painter.drawEllipse(x + shift[0] - r, y + shift[1] - r, 2 * r, 2 * r)
            painter.end()  # Finish painting
        else:
            for i, g in enumerate(group):
                rgb_color = color_palette[g]

                pen = QPen(QColor(*rgb_color))
                pen.setWidth(width)

                painter = QPainter(pixmap)
                painter.setPen(pen)

                radius = r + width * i
                painter.drawEllipse(
                    x + shift[0] - radius,
                    y + shift[1] - radius,
                    2 * radius,
                    2 * radius,
                )
                painter.end()  # Finish painting
        self.pixmap_item.setPixmap(pixmap)
        self.adjustImage()  # Adjust the image again to maintain fit

    def highlight_circle(self, circle, group=None, shift=None):
        x, y, r = circle
        self.highlight_circle_by_coordinates(x, y, r, group, shift)

    def resizeEvent(self, event):
        super().resizeEvent(event)  # Call the base class implementation
        self.adjustImage()  # Maintain aspect ratio on resize


class SpotDetectionUi:
    def __init__(self, path, parent=None):
        self.parent = parent
        self.image_path = path
        self.widget = QWidget()
        self.layout = QVBoxLayout()  # Create a QVBoxLayout for self.widget
        self.widget.setLayout(self.layout)  # Set the layout for self.widget
        self.inner_widget = QWidget()
        self.createActions()
        self.createToolbar()
        self.layout.addWidget(self.inner_widget)
        self.createHoughParameters()
        self.inner_layout = QHBoxLayout()
        self.inner_widget.setLayout(self.inner_layout)
        self.side_bar_layout = QVBoxLayout()
        # No circles detected yet
        self.circles_drawn = False
        # No clicks registered yet
        self.first_click = True
        self.initializeCircleDetection()
        self.createInteractiveImage()
        self.inner_layout.addLayout(self.side_bar_layout, stretch=1)
        self.createSpotPushButtons()
        self.createLayout()
        self.interactive_image.displayRepresentativeImage()
        self.detect_circles()

        self.current_group = 0

    def cleanup(self):
        """Clean up resources held by this object."""

        # Clean up the interactive image if it has a cleanup method
        if hasattr(self.interactive_image, "cleanup"):
            self.interactive_image.cleanup()
        self.interactive_image = None

        # Clean up the inner widget and its layout
        if self.inner_widget is not None:
            if self.inner_layout is not None:
                QWidget().setLayout(self.inner_layout)  # Break the layout reference
                self.inner_layout = None
            self.inner_widget.deleteLater()  # Safely delete the inner widget
            self.inner_widget = None

        # Clean up the main widget and layout
        if self.widget is not None:
            if self.layout is not None:
                QWidget().setLayout(self.layout)  # Break the layout reference
                self.layout = None
            self.widget.deleteLater()  # Safely delete the widget
            self.widget = None

        # Optionally: Break the reference to the parent
        self.parent = None

        # Set other attributes to None to ensure no references are held
        self.circles_drawn = False
        self.first_click = True

    def mousePressInImage(self, x, y):
        # TODO: There are some problems if the aspect ratio of the image is incorrect.
        if self.toggle_add_or_remove_circles_action.isChecked():
            self.add_or_remove_circle(x, y)
        elif self.toggle_measure_min_dist_action.isChecked():
            self.measure_min_dist(x, y)
        elif self.toggle_measure_spot_size_action.isChecked():
            self.measure_radius(x, y)
        elif self.toggle_move_spot_action.isChecked():
            self.select_spot(x, y)
        elif self.toggle_scale_radius_action.isChecked():
            self.select_spot(x, y)
        elif self.toggle_select_spots_action.isChecked():
            self.select_spot(x, y)

    def createActions(self):
        # Spot detection actions
        self.toggle_add_or_remove_circles_action = QAction(
            "Add or Remove Circles", self.parent
        )
        self.toggle_add_or_remove_circles_action.setStatusTip("Add or Remove Circles")
        self.toggle_add_or_remove_circles_action.setCheckable(True)

        self.toggle_measure_min_dist_action = QAction("Measure min Dist", self.parent)
        self.toggle_measure_min_dist_action.setStatusTip(
            "Measure minmal distance between two center points."
        )
        self.toggle_measure_min_dist_action.setCheckable(True)

        self.toggle_measure_spot_size_action = QAction("Measure spot size", self.parent)
        self.toggle_measure_spot_size_action.setStatusTip(
            "Measure spot size (diameter of one spot)."
        )
        self.toggle_measure_spot_size_action.setCheckable(True)

        self.toggle_move_spot_action = QAction("Move spot", self.parent)
        self.toggle_move_spot_action.setStatusTip("Move spot")
        self.toggle_move_spot_action.setCheckable(True)

        self.toggle_scale_radius_action = QAction("Scale radius", self.parent)
        self.toggle_scale_radius_action.setStatusTip("Scale Radius.")
        self.toggle_scale_radius_action.setCheckable(True)

        self.toggle_select_spots_action = QAction("Select Spots")
        self.toggle_select_spots_action.setStatusTip("Select spots for analysis.")
        self.toggle_select_spots_action.setCheckable(True)

        # Add the actions to a QActionGroup so that only one can be checked at any given time

        spot_detection_action_group = QActionGroup(self.parent)
        spot_detection_action_group.addAction(self.toggle_add_or_remove_circles_action)
        spot_detection_action_group.addAction(self.toggle_measure_min_dist_action)
        spot_detection_action_group.addAction(self.toggle_measure_spot_size_action)
        spot_detection_action_group.addAction(self.toggle_move_spot_action)
        spot_detection_action_group.addAction(self.toggle_scale_radius_action)
        spot_detection_action_group.addAction(self.toggle_select_spots_action)

        spot_detection_action_group.triggered.connect(self.spot_detection_tool_toggled)

        self.current_action = None

    def createToolbar(self):
        self.toolbar = QToolBar("Spot Detection Toolbar")
        self.layout.addWidget(self.toolbar)
        self.toolbar.addAction(self.toggle_add_or_remove_circles_action)
        self.toolbar.addAction(self.toggle_measure_min_dist_action)
        self.toolbar.addAction(self.toggle_measure_spot_size_action)
        self.toolbar.addAction(self.toggle_move_spot_action)
        self.toolbar.addAction(self.toggle_scale_radius_action)
        self.toolbar.addAction(self.toggle_select_spots_action)

    def createInteractiveImage(self):
        # Display an image
        self.interactive_image = InteractiveImage(self, self.circles)
        # self.interactive_image.setScaledContents(True)
        self.interactive_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.inner_layout.addWidget(self.interactive_image, stretch=3)
        self.interactive_image.setFocusPolicy(Qt.StrongFocus)
        self.folder_name = self.image_path.split("/")[-1]

    def createHoughParameters(self):

        self.hough_dp_label = QLabel("dp")
        self.hough_dp_param = QDoubleSpinBox()
        self.hough_dp_param.setValue(1.2)
        self.hough_dp_param.setSingleStep(0.1)

        self.hough_min_dist_label = QLabel("Min Dist")
        self.hough_min_dist_param = QSpinBox()
        self.hough_min_dist_param.setRange(0, 500)
        self.hough_min_dist_param.setValue(30)
        self.hough_min_dist_param.setSingleStep(1)

        self.hough_1_label = QLabel("Param 1")
        self.hough_1_param = QSpinBox()
        self.hough_1_param.setValue(13)
        self.hough_1_param.setSingleStep(1)

        self.hough_2_label = QLabel("Param 2")
        self.hough_2_param = QSpinBox()
        self.hough_2_param.setValue(20)
        self.hough_2_param.setSingleStep(1)

        self.hough_min_radius_label = QLabel("Min Radius")
        self.hough_min_radius_param = QSpinBox()
        self.hough_min_radius_param.setValue(8)
        self.hough_min_radius_param.setSingleStep(1)

        self.hough_max_radius_label = QLabel("Max Radius")
        self.hough_max_radius_param = QSpinBox()
        self.hough_max_radius_param.setValue(10)
        self.hough_max_radius_param.setSingleStep(1)

    def createSpotPushButtons(self):
        # Detect circles functionality
        self.detect_circles_button = QPushButton("Detect Circles")
        self.detect_circles_button.clicked.connect(self.detect_circles)

        # Calculate extinction
        self.calculate_extinction_button = QPushButton("Calculate Extinction")
        self.calculate_extinction_button.clicked.connect(
            self.calculate_extinction_button_clicked
        )

        self.deselect_spots_button = QPushButton("Deselect Spots")
        self.deselect_spots_button.clicked.connect(
            self.interactive_image.deselect_spots
        )

    def create_spot_selection_buttons(self):
        self.select_all_spots_button = QPushButton("Select All Spots")
        self.select_all_spots_button.clicked.connect(self.selectAllSpots)

        self.deselect_spots_button = QPushButton("Deselect All Spots")
        self.deselect_spots_button.clicked.connect(self.deselectSpots)

        self.add_group_button = QPushButton("Add New Group")
        self.add_group_button.clicked.connect(self.addGroup)

    def update_radii_for_extinction(self):
        spot_radius = (
            self.hough_min_radius_param.value() + self.hough_max_radius_param.value()
        ) * 0.5
        # Update the radii for extinction calculations
        self.background_inner_radius_param.setValue(int(spot_radius))
        self.background_outer_radius_param.setValue(int(spot_radius * 2))
        self.inner_radius_param.setValue(int(spot_radius * 0.5))

    def extinction_boundary_buttons(self):
        # Sets radii of the extinction calculations
        self.inner_radius_label = QLabel("Foreground Radius")
        self.inner_radius_label.setToolTip("Set the foreground radius in px.")
        self.inner_radius_param = QSpinBox()
        color = time_color_palette[0]
        self.inner_radius_param.setStyleSheet(
            f"background-color: rgba({color[0]}, {color[1]}, {color[2]}, 20);"
        )

        self.background_inner_radius_label = QLabel("Background Inner Radius")
        self.background_inner_radius_label.setToolTip(
            "Set the inner radius of the background computation in px."
        )
        self.background_inner_radius_param = QSpinBox()
        color = time_color_palette[1]
        self.background_inner_radius_param.setStyleSheet(
            f"background-color: rgba({color[0]}, {color[1]}, {color[2]}, 20);"
        )

        self.background_outer_radius_label = QLabel("Background Outer Radius")
        self.background_outer_radius_label.setToolTip(
            "Set the outer radius of the background computation in px."
        )
        self.background_outer_radius_param = QSpinBox()
        color = time_color_palette[1]
        self.background_outer_radius_param.setStyleSheet(
            f"background-color: rgba({color[0]}, {color[1]}, {color[2]}, 20);"
        )

        self.update_radii_for_extinction()
        self.inner_radius_param.valueChanged.connect(self.extinction_values_changed)
        self.inner_radius_param.setSingleStep(1)
        # self.inner_radius_param.setRange(0, 1)
        self.background_inner_radius_param.valueChanged.connect(
            self.extinction_values_changed
        )
        self.background_inner_radius_param.setSingleStep(1)
        # self.background_inner_radius_param.setRange(1, 1.5)
        self.background_outer_radius_param.valueChanged.connect(
            self.extinction_values_changed
        )
        self.background_outer_radius_param.setSingleStep(1)
        # self.background_outer_radius_param.setRange(1.5, 2)

        self.update_button = QPushButton("Update")
        self.update_button.clicked.connect(self.circles.compute_extinction)

    def draw_histogram(self):
        self.hist_widget.clear()
        # Ensure foreground and background data exist
        if self.circles.foreground is not None and self.circles.background is not None:
            # Flatten the arrays and concatenate them
            foreground_data = self.circles.foreground.flatten()
            background_data = self.circles.background.flatten()

            # Calculate histogram
            y, x = np.histogram(foreground_data, bins=50)

            brush = pg.mkBrush((*time_color_palette[0], 200))

            # Plot the histogram
            self.hist_widget.plot(x, y, stepMode=True, fillLevel=0, brush=brush)

            y, x = np.histogram(background_data, bins=50)

            brush = pg.mkBrush((*time_color_palette[1], 200))

            # Plot the histogram
            self.hist_widget.plot(x, y, stepMode=True, fillLevel=0, brush=brush)

    def extinction_values_changed(self):
        self.circles.extinction_bool = False

    def createLayout(self):
        # Create a group box for Hough Parameters
        hough_group_box = QGroupBox()
        # Make the group box checkable (this allows expanding and collapsing)
        hough_layout = QFormLayout()

        # Add Hough Parameters widgets
        hough_layout.addRow(self.hough_dp_label, self.hough_dp_param)
        hough_layout.addRow(self.hough_min_dist_label, self.hough_min_dist_param)
        hough_layout.addRow(self.hough_1_label, self.hough_1_param)
        hough_layout.addRow(self.hough_2_label, self.hough_2_param)
        hough_layout.addRow(self.hough_min_radius_label, self.hough_min_radius_param)
        hough_layout.addRow(self.hough_max_radius_label, self.hough_max_radius_param)

        hough_group_box.setLayout(hough_layout)

        # Add Push buttons button
        hough_layout.addWidget(self.detect_circles_button)
        # hough_layout.addWidget(self.calculate_extinction_button)
        # hough_layout.addWidget(self.deselect_spots_button)

        # Add Hough Parameters group box and Detect Circles button to the main layout

        # Create spot labeling group box
        spot_labeling_box = QGroupBox()
        self.spot_label_layout = QVBoxLayout()
        spot_labeling_box.setLayout(self.spot_label_layout)
        # self.side_bar_layout.addWidget(spot_labeling_box)

        self.create_spot_selection_buttons()

        self.spot_label_layout.addWidget(self.select_all_spots_button)
        self.spot_label_layout.addWidget(self.deselect_spots_button)
        self.spot_label_layout.addWidget(self.add_group_button)

        # Create counting logic
        self.group_count = 0

        self.spot_labels = QButtonGroup()
        self.spot_labels.setExclusive(True)

        # Flush buttons to the top
        self.spacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.spot_label_layout.addSpacerItem(self.spacer)

        # Add the first group
        self.addGroup()

        self.extinction_boundary_buttons()
        # Create a group box for extinction boundaries
        extinction_boundaries_box = QGroupBox()
        extinction_layout = QFormLayout()

        # Add extinction boundaries widgets
        extinction_layout.addRow(self.inner_radius_label, self.inner_radius_param)
        extinction_layout.addRow(
            self.background_inner_radius_label, self.background_inner_radius_param
        )
        extinction_layout.addRow(
            self.background_outer_radius_label, self.background_outer_radius_param
        )
        extinction_layout.addWidget(self.update_button)

        extinction_boundaries_box.setLayout(extinction_layout)
        # self.side_bar_layout.addWidget(extinction_boundaries_box)
        # Create a histogram plot using pyqtgraph
        self.hist_widget = pg.PlotWidget(background="w")
        self.hist_widget.setTitle(
            title="Foreground and Background Histogram", color="black"
        )

        self.hist_widget.setLabel("left", "Counts")  # Y-axis
        self.hist_widget.setLabel("bottom", "Intensity", units="a.u.")  # X-axis

        toolbox = QToolBox()
        toolbox.addItem(hough_group_box, "Hough Parameters")
        toolbox.addItem(spot_labeling_box, "Spot Labeling")
        toolbox.addItem(extinction_boundaries_box, "Extinction Boundaries")

        self.side_bar_layout.addWidget(toolbox, stretch=1)

        self.side_bar_layout.addWidget(self.hist_widget, stretch=1)

    def addGroup(self):
        self.group_count += 1
        group_name = f"Group {self.group_count}"
        index = self.group_count - 1

        # Cycle through the colors if more than the predefined colors are used
        color = color_palette[(index) % len(color_palette)]
        color_string = f"rgb({color[0]}, {color[1]}, {color[2]})"

        # Create the new radio button
        new_radio_button = QRadioButton(group_name)
        new_radio_button.setStyleSheet(
            f"""
            QRadioButton {{
                border: 2px solid {color_string};
                border-radius: 5px;
                padding: 5px;
            }}
        """
        )
        new_radio_button.setProperty("index", index)
        new_radio_button.toggled.connect(self.setCurrentGroup)
        new_radio_button.setChecked(True)
        self.spot_labels.addButton(new_radio_button)

        # Ensure that the spacer is always the last item in the group
        self.spot_label_layout.removeItem(self.spacer)

        # Add the new radio button to the layout and store it
        self.spot_label_layout.addWidget(new_radio_button)

        self.spot_label_layout.addSpacerItem(self.spacer)

    def setCurrentGroup(self):
        for button in self.spot_labels.buttons():
            if button.isChecked():
                self.current_group = button.property("index")

    def deselectSpots(self):
        self.interactive_image.deselect_spots()
        # self.parent.extinction_ui.updateSpotIndices(
        #     spot_indices=np.array([]), spot_labels=np.array([])
        # )
        # self.parent.time_series.updateSpotIndices(spot_indices=None, spot_labels=None)

    def selectAllSpots(self):
        self.interactive_image.select_all_spots()
        # self.updateSpotIndices()

    def updateSpotIndices(self):
        self.parent.extinction_ui.updateSpotIndices(
            spot_indices=self.circles.getSelectedSpotIndices(),
            spot_labels=self.circles.getSelectedSpotLabels(),
        )
        self.parent.time_series.updateSpotIndices(
            spot_indices=self.circles.getSelectedSpotIndices(),
            spot_labels=self.circles.getSelectedSpotLabels(),
        )
        return

    # Initialize circle detection class
    def initializeCircleDetection(self):
        dp = self.hough_dp_param.value()
        min_dist = self.hough_min_dist_param.value()
        param1 = self.hough_1_param.value()
        param2 = self.hough_2_param.value()
        min_radius = self.hough_min_radius_param.value()
        max_radius = self.hough_max_radius_param.value()
        self.circles = Circles(
            self.image_path,
            dp,
            min_dist,
            param1,
            param2,
            min_radius,
            max_radius,
            self,
        )

    def keyPressEvent(self, event: QKeyEvent):
        if self.toggle_move_spot_action.isChecked():
            key = event.key()
            if key == Qt.Key_Left:
                self.circles.selected_spot[0] -= 1
            elif key == Qt.Key_Right:
                self.circles.selected_spot[0] += 1
            elif key == Qt.Key_Down:
                self.circles.selected_spot[1] += 1
            elif key == Qt.Key_Up:
                self.circles.selected_spot[1] -= 1
        elif self.toggle_scale_radius_action.isChecked():
            key = event.key()
            if key == Qt.Key_Left:
                self.circles.selected_spot[2] -= 1
            elif key == Qt.Key_Right:
                self.circles.selected_spot[2] += 1
        self.circles.draw_circles()

    def spot_detection_tool_toggled(self, action):
        self.deselectSpots()
        self.start_point = None
        """ Custom logic to allow unchecking an action when clicked again """
        if self.current_action == action:
            # If the same action is clicked again, uncheck it
            action.setChecked(False)
            self.current_action = None
        else:
            # Otherwise, uncheck the previous one and set the new one
            if self.current_action:
                self.current_action.setChecked(False)

            self.current_action = action
            action.setChecked(True)
        self.deselectSpots()

    def add_or_remove_circle(self, x, y):
        pixmap = self.circles.add_or_remove_circle(x, y)
        self.interactive_image.setPixmap(pixmap)
        self.detected_circles_pixmap = pixmap

    def select_spot(self, x, y):
        self.interactive_image.select_circle(x, y)

    def detect_circles(self):
        dp = self.hough_dp_param.value()
        min_dist = self.hough_min_dist_param.value()
        param1 = self.hough_1_param.value()
        param2 = self.hough_2_param.value()
        min_radius = self.hough_min_radius_param.value()
        max_radius = self.hough_max_radius_param.value()

        self.circles.update_parameters(
            dp,
            min_dist,
            param1,
            param2,
            min_radius,
            max_radius,
        )
        self.update_radii_for_extinction()
        pixmap = self.circles.draw_circles()
        self.detected_circles_pixmap = pixmap
        self.interactive_image.setPixmap(pixmap)
        self.circles_drawn = True

    def calculate_extinction_button_clicked(self):
        pixmap = self.circles.compute_extinction()

    def measure_min_dist(self, x, y):
        if self.start_point is None:
            self.start_point = QPoint(x, y)
            self.interactive_image.setPixmap(self.circles.get_representative_image())
        else:
            self.end_point = QPoint(x, y)
            self.measured_dist = (
                0.9
                * (
                    (self.end_point.x() - self.start_point.x()) ** 2
                    + (self.end_point.y() - self.start_point.y()) ** 2
                )
                ** 0.5
            )
            self.hough_min_dist_param.setValue(self.measured_dist)
            self.circles.update_min_dist(self.measured_dist)
            self.draw_measurement()
            self.start_point = None

    def measure_radius(self, x, y):
        if self.start_point is None:
            self.start_point = QPoint(x, y)
            self.interactive_image.setPixmap(self.circles.get_representative_image())
        else:
            self.end_point = QPoint(x, y)
            self.measured_dist = 0.5 * (
                (
                    (self.end_point.x() - self.start_point.x()) ** 2
                    + (self.end_point.y() - self.start_point.y()) ** 2
                )
                ** 0.5
            )
            self.hough_min_radius_param.setValue(0.85 * self.measured_dist)
            self.circles.update_min_radius(0.85 * self.measured_dist)
            self.hough_max_radius_param.setValue(1.15 * self.measured_dist)
            self.circles.update_max_radius(1.15 * self.measured_dist)
            self.draw_measurement()
            self.start_point = None

    def draw_measurement(self):
        self.interactive_image.draw_measurement(
            self.start_point, self.end_point, self.measured_dist
        )
