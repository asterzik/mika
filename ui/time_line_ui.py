import pyqtgraph as pg
import numpy as np

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
)


class TimeLine:
    def __init__(self, parent=None):
        self.parent = parent
        self.widget = QWidget()
        self.layout = QHBoxLayout()
        self.widget.setLayout(self.layout)
        self.plot_widget = pg.PlotWidget()
        self.layout.addWidget(self.plot_widget)
        self.button_layout = QVBoxLayout()
        self.layout.addLayout(self.button_layout)
        self.plot_widget.setBackground("w")
        self.plot_widget.setTitle("Time Line", color="black")
        self.color_offset = 10
        self.cur_color = None

        self.time_steps = None
        self.time_ranges = []  # List to store the time ranges
        self.first = True

        self.buttons()

        # Disable panning and zooming
        self.view_box = self.plot_widget.getViewBox()
        self.view_box.setMouseEnabled(x=False, y=False)

    def cleanup(self):
        """Clean up resources held by this object."""

        # Clean up plot_widget
        if self.plot_widget is not None:
            self.plot_widget.clear()  # Clear any data in the plot
            self.plot_widget.deleteLater()  # Schedule it for deletion
            self.plot_widget = None

        # Clean up the view box if necessary
        if self.view_box is not None:
            self.view_box.setParentItem(None)  # Break any parent-child link
            self.view_box = None

        # Clean up the main widget and layout
        if self.widget is not None:
            self.widget.deleteLater()  # Safely delete the widget
            self.widget = None

        # Optionally break reference to parent
        self.parent = None

        # Reset other attributes
        self.color_offset = None
        self.cur_color = None
        self.time_steps = None
        self.time_ranges = None
        self.first = None

    def buttons(self):
        self.add_range_button = QPushButton("Add New Time Range")
        self.add_range_button.clicked.connect(self.addRange)
        self.button_layout.addWidget(self.add_range_button)

        # Create counting logic
        self.range_count = 0

        self.range_labels = QButtonGroup()
        self.range_labels.setExclusive(True)

        # Flush buttons to the top
        self.spacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.button_layout.addSpacerItem(self.spacer)

    def addRange(self, first):
        self.range_count += 1
        range_name = f"Group {self.range_count}"
        index = self.range_count - 1

        # Cycle through the colors if more than the predefined colors are used
        color = time_color_palette[(index) % len(color_palette)]
        color_string = f"rgb({color[0]}, {color[1]}, {color[2]})"

        # Create the new radio button
        new_radio_button = QRadioButton(range_name)
        new_radio_button.setStyleSheet(
            f"""
            QRadioButton {{
                border: 2px solid {color_string};
                border-radius: 5px;
                padding: 5px;
            }}
        """
        )
        self.current_range = index
        new_radio_button.setProperty("index", index)
        self.range_labels.addButton(new_radio_button)

        # Ensure that the spacer is always the last item in the group
        self.button_layout.removeItem(self.spacer)

        # Add the new radio button to the layout and store it
        self.button_layout.addWidget(new_radio_button)

        self.button_layout.addSpacerItem(self.spacer)

        # Highlight the area and store it in the list
        xmin = 0.7
        xmax = 1.3
        if first:
            xmin = self.time_steps[0] - 0.1
            xmax = self.time_steps[len(self.time_steps) - 1] + 0.1
        self.highlightXArea(xmin, xmax)

    def draw(self):
        self.plot_widget.clear()
        self.time_steps, _ = self.parent.extinction_ui.get_time_series()
        self.plot_widget.plot(
            self.time_steps,
            np.ones_like(self.time_steps),
            pen="black",
            symbol="o",
            symbolBrush="black",
        )
        xmin = self.time_steps[0] - 0.1
        xmax = self.time_steps[len(self.time_steps) - 1] + 0.1
        self.view_box.setLimits(xMin=xmin, xMax=xmax)
        self.plot_widget.getPlotItem().hideAxis("left")
        # Add the first group
        if self.first:
            self.addRange(True)
            self.first = False
        self.time_region.sigRegionChangeFinished.connect(self.updateTimeValues)
        self.getTimeValues()

    def highlightXArea(self, xmin, xmax):
        color = QColor(*time_color_palette[(self.current_range) % len(color_palette)])
        color.setAlpha(60)

        # Create a new LinearRegionItem (the time range)
        self.time_region = pg.LinearRegionItem(
            orientation="vertical", movable=True, brush=color
        )
        self.time_region.setRegion((xmin, xmax))
        self.plot_widget.addItem(self.time_region)

        # Store the created time range in the list
        self.time_ranges.append(self.time_region)

        self.time_region.sigRegionChangeFinished.connect(self.updateTimeValues)

        # Call updateTimeValues to update the values for all regions if additional ranges are added
        if self.range_count > 1:
            {self.updateTimeValues()}

    # def getAllTimeRanges(self):
    #     """Returns a list of tuples with the (xmin, xmax) values of all time ranges."""
    #     return [region.getRegion() for region in self.time_ranges]

    def getTimeValues(self):
        all_indices = []
        all_labels = []

        # Loop through all time regions in self.time_ranges
        for i, time_region in enumerate(self.time_ranges):
            # Get the current region's (xmin, xmax)
            current_range = time_region.getRegion()
            # Create a mask for the current region
            mask = (self.time_steps > current_range[0]) & (
                self.time_steps < current_range[1]
            )
            # Get the indices within the current region
            indices = np.where(mask)[0]

            # Append the indices and their corresponding region label
            for idx in indices:
                all_indices.append(idx)
                all_labels.append(i)

        # Get unique indices and their corresponding labels
        unique_indices, unique_indices_idx = np.unique(all_indices, return_index=True)

        # Gather the corresponding labels for unique indices
        unique_labels = [all_labels[j] for j in unique_indices_idx]

        return unique_indices, unique_labels

    def updateTimeValues(self):
        indices, labels = self.getTimeValues()
        self.parent.time_series.updateTimeIndices(indices, labels)
        self.parent.extinction_ui.updateTimeIndices(
            time_indices=indices, time_labels=labels
        )
