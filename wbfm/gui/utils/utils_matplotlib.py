# From https://stackoverflow.com/questions/48140576/matplotlib-toolbar-in-a-pyqt5-application
import random

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QSizePolicy
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


class PlotQWidget(QWidget):
    def __init__(self, *args, **kwargs):
        QWidget.__init__(self, *args, **kwargs)
        self.setLayout(QVBoxLayout())
        self.canvas = PlotCanvas(self, width=10, height=3)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.layout().addWidget(self.toolbar)
        self.layout().addWidget(self.canvas)

    def draw(self):
        self.canvas.draw()


class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=10, height=8, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    # def plot(self):
    #     data = [random.random() for i in range(250)]
    #     ax = self.figure.add_subplot(111)
    #     ax.plot(data, 'r-', linewidth = 0.5)
    #     ax.set_title('PyQt Matplotlib Example')
    #     self.draw()


def get_twin_axis(ax, axis='x'):
    # From: https://stackoverflow.com/questions/36209575/how-to-detect-if-a-twin-axis-has-been-generated-for-a-matplotlib-axis
    assert axis in ("x", "y")
    siblings = getattr(ax, f"get_shared_{axis}_axes")().get_siblings(ax)
    for sibling in siblings:
        if sibling.bbox.bounds == ax.bbox.bounds and sibling is not ax:
            return sibling
    return None
