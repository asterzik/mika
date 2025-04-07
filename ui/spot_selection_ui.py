import os
from PySide6.QtWidgets import (
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
    QRadioButton,
    QSpinBox,
    QGroupBox,
    QSlider,
    QSizePolicy,
    QGraphicsScene,
    QGraphicsPixmapItem,
    QFileDialog,
)
from PySide6.QtCore import Qt

from PySide6.QtCore import Qt

from misc.graphics_view import CustomGraphicsView


class SpotSelectionUi:
    def __init__(self, path, parent=None):
        self.parent = parent
        self.circles = self.parent.spot_ui.circles
        self.widget = QWidget()
        self.layout0 = QHBoxLayout()

        # Create a CustomGraphicsView and QGraphicsScene
        self.graphics_view = CustomGraphicsView(self)
        self.graphics_scene = QGraphicsScene(self.graphics_view)
        self.graphics_view.setScene(self.graphics_scene)

        # Extract the pixmap from the interactive image
        self.interactive_image_pixmap = (
            self.parent.spot_ui.interactive_image.pixmap_item.pixmap()
        )

        # Create a QGraphicsPixmapItem and set the pixmap
        self.pixmap_item = QGraphicsPixmapItem(self.interactive_image_pixmap)
        self.graphics_scene.addItem(self.pixmap_item)

        # Set the size policy and scaling mode
        self.graphics_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.graphics_view.fitInView(self.pixmap_item, Qt.KeepAspectRatio)

        self.widget.setLayout(self.layout0)
        # Add the CustomGraphicsView to the layout
        self.layout0.addWidget(self.graphics_view, stretch=3)

        self.image_selection_layout = QVBoxLayout()

        self.layout0.addLayout(self.image_selection_layout, stretch=1)

        # self.export_button = QPushButton("Export Spot Image")
        # self.export_button.clicked.connect(self.export)
        # self.image_selection_layout.addWidget(self.export_button)

        self.build_image_selection_group()

    def export(self):
        proposed_filename = os.path.join(os.curdir, "spot_selection.png")
        filename, _ = QFileDialog.getSaveFileName(
            self.widget,
            "Export Image",
            proposed_filename,
            "Image Files (*.png *.jpg *.jpeg)",
        )
        if filename:
            self.pixmap_item.pixmap().save(filename)

    def update_image(self):
        # Retrieve the updated pixmap from the external source
        updated_pixmap = self.parent.spot_ui.interactive_image.pixmap_item.pixmap()

        # Update the QGraphicsPixmapItem with the new pixmap
        self.pixmap_item.setPixmap(updated_pixmap)

        # Call fitInView to ensure the new pixmap is properly scaled and displayed
        self.graphics_view.fitInView(self.pixmap_item, Qt.KeepAspectRatio)

    def cleanup(self):
        """Clean up resources held by this object."""

        # Clean up the circles if it has a cleanup method
        if self.circles and hasattr(self.circles, "cleanup"):
            self.circles.cleanup()
        self.circles = None

        # Clean up the widget
        if self.widget is not None:
            self.widget.deleteLater()  # Safely delete the widget
            self.widget = None

        # Optional: Break reference to parent
        self.parent = None

    def build_image_selection_group(self):
        # Create a QGroupBox to group the radio buttons
        radio_group_box = QGroupBox("Image Selection")

        # Create a vertical layout for the radio buttons inside the group box
        radio_layout = QHBoxLayout()

        # Create radio buttons
        self.mean_image_radio = QRadioButton("Highest Contrast")
        self.selected_image_radio = QRadioButton("Custom")

        # Add radio buttons to the group box layout
        radio_layout.addWidget(self.mean_image_radio)
        radio_layout.addWidget(self.selected_image_radio)

        # Set layout for the group box
        radio_group_box.setLayout(radio_layout)

        # Add the group box to the main button layout
        self.image_selection_layout.addWidget(radio_group_box)

        # Set default selection (optional)
        self.mean_image_radio.setChecked(True)

        self.mean_image_radio.toggled.connect(self.displayMeanImage)
        self.selected_image_radio.toggled.connect(self.displaySelectedImage)

        # Selection opportunity for Frame
        frame_group = QGroupBox("Frame")
        frame_group_layout = QVBoxLayout()
        self.frame_spin_box = QSpinBox()
        frames = self.circles.get_frames()
        self.frame_spin_box.setRange(frames[0], frames[len(frames) - 1])
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setRange(frames[0], frames[len(frames) - 1])
        self.frame_slider.setTickInterval(1)
        self.frame_slider.setTickPosition(QSlider.TicksBelow)

        def update_frame_spinbox(index):
            self.frame_spin_box.setValue(index)
            self.redrawImage()

        def update_frame_slider(value):
            self.frame_slider.setValue(value)
            self.redrawImage()

        # Synchronize the spinbox and slider
        self.frame_slider.valueChanged.connect(update_frame_spinbox)
        self.frame_spin_box.valueChanged.connect(update_frame_slider)

        frame_group_layout.addWidget(self.frame_spin_box)
        frame_group_layout.addWidget(self.frame_slider)
        frame_group.setLayout(frame_group_layout)

        self.image_selection_layout.addWidget(frame_group)

        # Selection for wavelength

        wavelength_group = QGroupBox("Wavelength")
        wavelength_group_layout = QVBoxLayout()
        self.wl_spin_box = QSpinBox()
        wavelengths = self.circles.get_wavelengths()
        self.wl_spin_box.setRange(wavelengths[0], wavelengths[len(wavelengths) - 1])
        # TODO: change selection method if the distances between the wavelengths aren't always the same
        step_size = wavelengths[1] - wavelengths[0]
        self.wl_spin_box.setSingleStep(step_size)
        self.wl_slider = QSlider(Qt.Horizontal)
        self.wl_slider.setRange(0, len(wavelengths) - 1)
        self.wl_slider.setSingleStep(1)
        self.wl_slider.setTickInterval(1)
        self.wl_slider.setTickPosition(QSlider.TicksBelow)

        def update_wl_spinbox(value):
            self.wl_spin_box.setValue(wavelengths[0] + value * step_size)
            self.redrawImage()

        def update_wl_slider(value):
            self.wl_slider.setValue((value - wavelengths[0]) // step_size)
            self.redrawImage()

        # Synchronize the spinbox and slider
        self.wl_slider.valueChanged.connect(update_wl_spinbox)
        self.wl_spin_box.valueChanged.connect(update_wl_slider)

        wavelength_group_layout.addWidget(self.wl_spin_box)
        wavelength_group_layout.addWidget(self.wl_slider)

        wavelength_group.setLayout(wavelength_group_layout)

        # Add the group box to the main layout
        self.image_selection_layout.addWidget(wavelength_group)

    def redrawImage(self):
        if self.mean_image_radio.isChecked():
            self.displayMeanImage()
        else:
            self.displaySelectedImage()

    def displayMeanImage(self):
        self.parent.spot_ui.interactive_image.displayRepresentativeImage()
        self.circles.highlight_selected(self.parent.spot_ui.interactive_image)
        self.update_image()
        return

    def displaySelectedImage(self):
        frame = self.frame_spin_box.value()
        wavelength = self.wl_spin_box.value()
        pixmap = self.circles.get_image(wavelength, frame)
        shift = self.circles.get_shift(wavelength)
        self.parent.spot_ui.interactive_image.setPixmap(pixmap)
        self.circles.highlight_selected(
            self.parent.spot_ui.interactive_image, shift=shift
        )
        self.update_image()

        return
