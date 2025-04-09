from PySide6.QtWidgets import QGraphicsView, QMenu
from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QAction


class SpotDetectionView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__()
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.parent_ui = parent

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            self.parent_ui.export()  # Call the export method from the parent UI
        else:
            super().mousePressEvent(event)


class SpotMetricsView(QGraphicsView):
    def __init__(self, results_view):
        super().__init__()
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.results_view = results_view

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)

    def contextMenuEvent(self, event):
        print("Context menu event triggered")
        menu = QMenu(self)

        save_image_action = QAction("Save Image + Legend", self)
        export_csv_action = QAction("Export Data to CSV", self)

        save_image_action.triggered.connect(self.results_view.saveImageAndLegend)
        export_csv_action.triggered.connect(self.results_view.exportDataToCSV)

        menu.addAction(save_image_action)
        menu.addAction(export_csv_action)

        menu.exec_(event.globalPos())
