from PySide6.QtWidgets import QWidget, QHBoxLayout


class ResultsView:
    def __init__(self, parent=None):
        self.parent = parent
        self.widget = QWidget()
        self.layout = QHBoxLayout()
        self.widget.setLayout(self.layout)
