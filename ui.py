from image import Image
from filter import medianFilter, meanFilter, gaussFilter, bilateralFilter, maxFilter
from tracking import Tracking

import sys
from PyQt5.QtCore import pyqtSlot, pyqtSignal, QThread
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QDialog, QHBoxLayout, QVBoxLayout, QGroupBox
from PyQt5 import QtCore, QtGui, QtWidgets
import skimage.io
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QPixmap, QColor
from qimage2ndarray import array2qimage, gray2qimage
import skimage
from skimage.restoration import denoise_bilateral
from skimage import morphology
import math
from openpyxl import Workbook
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np


class AppWidget(QtWidgets.QMainWindow):
    #loading UI and initial parameters
    def __init__(self):
        QWidget.__init__(self)
        self.resize(1920, 1080)
        self.setWindowTitle("Object Tracking")

        self.images = []
        self.index = None
        self.neighborhoodSize = None
        self.filter = "Median Filter"
        self.currentImage = []
        self.segmentationTech = "Threshold"
        self.currentFiltImg = []
        self.running_thread = None
        self.currentState = "Contrast"
        self.colors = ('blue', 'cyan', 'darkorange', 'indigo',
                 'magenta', 'pink', 'red', 'yellow', 'yellowgreen')
        self.labels = ["Label","Color","CentroiX","CentroidY","Area","Orientation", "Ellongation"]
        self.uniqueLabel = []
        self.labelCount = [0] * 100
        self.pozX = [[] for x in range(100)]
        self.pozY = [[] for x in range(100)]
        self.orientation = [[] for x in range(100)]
        self.ellongation = [[] for x in range(100)]
        self.pixmap = None
        self.changedLabels = {}

        self.lowThresh = None
        self.highThresh = None


        self.neigbrhoodSize = None
        self.threshold = None
        self.kernel = None
        self.sigma = None
        self.sigmaColor = None
        self.sigmaSpatial = None
        self.manThreshold = None
        self.sigmaCanny = None
        self.imgFiltNorm = None
        self.lowThresh = None
        self.highThresh = None
        self.size = None
        self.sizeKernel = None
        self.autoMethod = None

        self.objectProperties = []

        ## Contrast sliders ##

        self.contrastGroupBox = QtWidgets.QGroupBox("Contrast Stretchng", self)
        self.contrastGroupBox.setGeometry(QtCore.QRect(1030, 900, 870, 105))
        self.contrastGroupBox.setObjectName("contrastGroupBox")
        self.contrastGroupBox.setEnabled(False)

        self.verticalLayoutWidget = QtWidgets.QWidget(self.contrastGroupBox)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(5, 5, 860, 95))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")

        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")

        self.blackSlider = QtWidgets.QSlider(self.verticalLayoutWidget)
        self.blackSlider.setOrientation(QtCore.Qt.Horizontal)
        self.blackSlider.setObjectName("blackSlider")
        self.verticalLayout.addWidget(self.blackSlider)
        self.blackSlider.sliderReleased.connect(self.contrast)
        self.blackSlider.setMaximum(4096)

        self.whiteSlider = QtWidgets.QSlider(self.verticalLayoutWidget)
        self.whiteSlider.setOrientation(QtCore.Qt.Horizontal)
        self.whiteSlider.setObjectName("whiteSlider")
        self.verticalLayout.addWidget(self.whiteSlider)
        self.whiteSlider.sliderReleased.connect(self.contrast)
        self.whiteSlider.setMaximum(4096)
        self.whiteSlider.setProperty("value", 4096)

        ### Background Groupbox and components ##

        self.backgroundGroupBox = QtWidgets.QGroupBox("Background", self)
        self.backgroundGroupBox.setGeometry(QtCore.QRect(245, 25, 470, 100))
        self.backgroundGroupBox.setObjectName("backgroundGroupBox")
        self.backgroundGroupBox.setEnabled(False)

        self.horizontBackgroundGB = QtWidgets.QWidget(
            self.backgroundGroupBox)
        self.horizontBackgroundGB.setGeometry(
            QtCore.QRect(10, 20, 450, 25))
        self.horizontBackgroundGB.setObjectName("horizontalLayoutWidget_2")

        self.horizontLayoutBackgroundGB = QtWidgets.QHBoxLayout(
            self.horizontBackgroundGB)
        self.horizontLayoutBackgroundGB.setContentsMargins(0, 0, 0, 0)
        self.horizontLayoutBackgroundGB.setObjectName(
            "horizontLayoutBackgroundGB")

        self.neigborhoodSizeLabel = QtWidgets.QLabel(
            "Neighborhood Size: ", self.horizontBackgroundGB)
        self.neigborhoodSizeLabel.move(10, 10)
        self.horizontLayoutBackgroundGB.addWidget(self.neigborhoodSizeLabel)

        self.onlyIntValidator = QtGui.QIntValidator(1, 999)

        self.neigborhoodSize = QtWidgets.QLineEdit(
            "5", self.horizontBackgroundGB)
        self.neigborhoodSize.setGeometry(QtCore.QRect(20, 10, 5, 50))
        self.horizontLayoutBackgroundGB.addWidget(self.neigborhoodSize)
        self.neigborhoodSize.setValidator(self.onlyIntValidator)

        self.medianBackgroundButton = QtWidgets.QPushButton(
            "Median Background", self.horizontBackgroundGB)
        self.medianBackgroundButton.setObjectName("medianBackgroundButton")
        self.horizontLayoutBackgroundGB.addWidget(self.medianBackgroundButton)
        self.medianBackgroundButton.clicked.connect(self.backgroundMedian)

        self.dilationButton = QtWidgets.QPushButton(
            "Dilation", self.horizontBackgroundGB)
        self.dilationButton.setObjectName("dilationButton")
        self.horizontLayoutBackgroundGB.addWidget(self.dilationButton)
        self.dilationButton.clicked.connect(self.dilation)
        self.dilationButton.setEnabled(False)

        self.meanBackgroundButton = QtWidgets.QPushButton(
            "Mean Background", self.horizontBackgroundGB)
        self.meanBackgroundButton.setObjectName("meanBackgroundButton")
        self.horizontLayoutBackgroundGB.addWidget(self.meanBackgroundButton)
        self.meanBackgroundButton.clicked.connect(self.backgroundMean)
        self.meanBackgroundButton.setEnabled(False)

        self.verticalLayoutWidget_2 = QtWidgets.QWidget(
            self.backgroundGroupBox)
        self.verticalLayoutWidget_2.setGeometry(QtCore.QRect(10, 40, 390, 71))
        self.verticalLayoutWidget_2.setObjectName("verticalLayoutWidget_2")

        self.verticalLayout_2 = QtWidgets.QVBoxLayout(
            self.verticalLayoutWidget_2)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")

        self.backgroundSlider = QtWidgets.QSlider(
            self.verticalLayoutWidget_2)
        self.backgroundSlider.setOrientation(QtCore.Qt.Horizontal)
        self.backgroundSlider.setObjectName("backgroundSlider")
        self.verticalLayout_2.addWidget(self.backgroundSlider)
        self.backgroundSlider.setEnabled(False)
        self.backgroundSlider.sliderReleased.connect(self.thresholdBackground)

        self.thresholdWindow = QtWidgets.QLineEdit(self.backgroundGroupBox)
        self.thresholdWindow.setGeometry(QtCore.QRect(410, 65, 50, 20))
        self.thresholdWindow.setObjectName("thresholdWindow")
        self.thresholdWindow.setEnabled(False)

        ## Image Label ##

        self.imageWindow = QtWidgets.QLabel(self)
        self.imageWindow.setGeometry(QtCore.QRect(1030, 25, 870, 870))
        self.imageWindow.setObjectName("imageWindow")
        self.imageWindow.setStyleSheet("border: 1px solid black;")

        self.positionLabel = QtWidgets.QLineEdit(self)
        self.positionLabel.setGeometry(QtCore.QRect(1035, 860, 180, 30))
        self.positionLabel.setObjectName("positionLabel")
        self.positionLabel.setStyleSheet("border: 1px solid black;")
        self.positionLabel.setEnabled(False)

        ## Paths list ##

        self.pathsList = QtWidgets.QListWidget(self)
        self.pathsList.setGeometry(QtCore.QRect(20, 30, 215, 465))
        self.pathsList.setObjectName("pathsList")
        self.pathsList.itemDoubleClicked.connect(self.pathPosition)
        self.pathsList.setEnabled(False)

        ## Table Widget and buttons ##

        self.objectPropertiesTable = QtWidgets.QTableWidget(self)
        self.objectPropertiesTable.setGeometry(QtCore.QRect(20, 500, 1000, 470))
        self.objectPropertiesTable.setObjectName("objectPropertiesTable")
        self.objectPropertiesTable.setColumnCount(0)
        self.objectPropertiesTable.setRowCount(0)
        self.objectPropertiesTable.setEnabled(False)
        self.objectPropertiesTable.setSelectionBehavior(QtWidgets.QTableWidget.SelectRows)

        self.tableButtonsGroupBox = QtWidgets.QGroupBox(self)
        self.tableButtonsGroupBox.setGeometry(
            QtCore.QRect(20, 975, 620, 35))
        self.tableButtonsGroupBox.setEnabled(False)
        self.tableButtonsGroupBox.setFlat(True)
        self.tableButtonsGroupBox.setObjectName("tableButtonsGroupBox")
        self.tableButtonsGroupBox.setStyleSheet("tableButtonsGroupBox {border: 0px}")

        self.horizontalLayoutWidget = QtWidgets.QWidget(self.tableButtonsGroupBox)
        self.horizontalLayoutWidget.setGeometry(
            QtCore.QRect(5, 5, 600, 25))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")

        self.horizontalLayout = QtWidgets.QHBoxLayout(
            self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")

        self.sortTableButton = QtWidgets.QPushButton(
            "Sort", self.horizontalLayoutWidget)
        self.sortTableButton.setObjectName("sortTableButton")
        self.horizontalLayout.addWidget(self.sortTableButton)
        self.sortTableButton.clicked.connect(self.sortTable)

        self.exportTableButton = QtWidgets.QPushButton(
            "Export", self.horizontalLayoutWidget)
        self.exportTableButton.setObjectName("exportTableButton")
        self.horizontalLayout.addWidget(self.exportTableButton)
        self.exportTableButton.clicked.connect(self.exportTable)

        self.deleteTableButton = QtWidgets.QPushButton(
            "Delete", self.horizontalLayoutWidget)
        self.deleteTableButton.setObjectName("deleteTableButton")
        self.horizontalLayout.addWidget(self.deleteTableButton)
        self.deleteTableButton.clicked.connect(self.deleteRow)
        self.deleteTableButton.setEnabled(False)

        self.filterTableButton = QtWidgets.QPushButton(
            "Filter", self.horizontalLayoutWidget)
        self.filterTableButton.setObjectName("filterTableButton")
        self.horizontalLayout.addWidget(self.filterTableButton)
        self.filterTableButton.clicked.connect(self.filterTable)
        self.filterTableButton.setEnabled(False)

        self.plotTableButton = QtWidgets.QPushButton(
            "Plot Trajectory", self.horizontalLayoutWidget)
        self.plotTableButton.setObjectName("plotTableButton")
        self.horizontalLayout.addWidget(self.plotTableButton)
        self.plotTableButton.clicked.connect(self.plotResutlts)
        self.plotTableButton.setEnabled(False)

        self.plotOrientationButton = QtWidgets.QPushButton(
            "Plot Orientation", self.horizontalLayoutWidget)
        self.plotOrientationButton.setObjectName("plotOrientationButton")
        self.horizontalLayout.addWidget(self.plotOrientationButton)
        self.plotOrientationButton.clicked.connect(self.plotOrientation)
        self.plotOrientationButton.setEnabled(False)

        ## Filter Group Box ##

        self.filterGroupBox = QtWidgets.QGroupBox("Filtering", self)
        self.filterGroupBox.setGeometry(QtCore.QRect(245, 130, 470, 130))
        self.filterGroupBox.setObjectName("filterGroupBox")
        self.filterGroupBox.setEnabled(False)

        self.filterList = QtWidgets.QListWidget(self.filterGroupBox)
        self.filterList.setGeometry(QtCore.QRect(10, 20, 120, 75))
        self.filterList.setObjectName("filterList")
        self.filterList.addItem('Median Filter')
        self.filterList.addItem('Mean Filter')
        self.filterList.addItem('Gauss Filter')
        self.filterList.addItem('Bilateral Filter')
        self.filterList.addItem('Maximum Filter')
        self.filterList.itemDoubleClicked.connect(self.filterPosition)

        self.medianFilterGroupBox = QtWidgets.QGroupBox(
            "Median Filter", self.filterGroupBox)
        self.medianFilterGroupBox.setGeometry(QtCore.QRect(140, 15, 200, 110))
        self.medianFilterGroupBox.setObjectName("medianFilterGroupBox")
        self.medianFilterGroupBox.setVisible(False)

        self.medianFilterKernelLabel = QtWidgets.QLabel(
            "Kernel shape: ", self.medianFilterGroupBox)
        self.medianFilterKernelLabel.move(10, 25)

        self.medianFilterKernelGeometry = QtWidgets.QComboBox(
            self.medianFilterGroupBox)
        self.medianFilterKernelGeometry.setGeometry(
            QtCore.QRect(80, 22, 80, 20))
        self.medianFilterKernelGeometry.addItem("Square")
        self.medianFilterKernelGeometry.addItem("Disk")

        self.medianFilterKernelSizeLabel = QtWidgets.QLabel(
            "Kernel size: ", self.medianFilterGroupBox)
        self.medianFilterKernelSizeLabel.move(10, 55)

        self.medianFilterKernelSize = QtWidgets.QLineEdit("5",
            self.medianFilterGroupBox)
        self.medianFilterKernelSize.setGeometry(QtCore.QRect(80, 52, 80, 20))

        self.meanFilterGroupBox = QtWidgets.QGroupBox(
            "Mean Filter", self.filterGroupBox)
        self.meanFilterGroupBox.setGeometry(QtCore.QRect(140, 15, 200, 110))
        self.meanFilterGroupBox.setObjectName("meanFilterGroupBox")
        self.meanFilterGroupBox.setVisible(False)

        self.meanFilterKernelSizeLabel = QtWidgets.QLabel(
            "Kernel size: ", self.meanFilterGroupBox)
        self.meanFilterKernelSizeLabel.move(10, 25)

        self.meanFilterKernelSize = QtWidgets.QLineEdit("5",
            self.meanFilterGroupBox)
        self.meanFilterKernelSize.setGeometry(QtCore.QRect(80, 22, 80, 20))

        self.gaussFilterGroupBox = QtWidgets.QGroupBox(
            "Gauss Filter", self.filterGroupBox)
        self.gaussFilterGroupBox.setGeometry(QtCore.QRect(140, 15, 200, 110))
        self.gaussFilterGroupBox.setObjectName("gaussFilterGroupBox")
        self.gaussFilterGroupBox.setVisible(False)

        self.gaussFilterSigmaValueLabel = QtWidgets.QLabel(
            "Sigma: ", self.gaussFilterGroupBox)
        self.gaussFilterSigmaValueLabel.move(10, 25)

        self.gaussFilterSigmaValue = QtWidgets.QLineEdit("2",
            self.gaussFilterGroupBox)
        self.gaussFilterSigmaValue.setGeometry(QtCore.QRect(80, 22, 80, 20))

        self.bilateralFilterGroupBox = QtWidgets.QGroupBox(
            "Bilateral Filter", self.filterGroupBox)
        self.bilateralFilterGroupBox.setGeometry(
            QtCore.QRect(140, 15, 200, 110))
        self.bilateralFilterGroupBox.setObjectName("bilateralFilterGroupBox")
        self.bilateralFilterGroupBox.setVisible(False)

        self.bilateralFilterSigmaCLabel = QtWidgets.QLabel(
            "Intesity Sigma: ", self.bilateralFilterGroupBox)
        self.bilateralFilterSigmaCLabel.move(10, 25)

        self.bilateralFilterSigmaCValue = QtWidgets.QLineEdit("0.2",
            self.bilateralFilterGroupBox)
        self.bilateralFilterSigmaCValue.setGeometry(
            QtCore.QRect(90, 22, 80, 20))

        self.bilateralFilterSigmaSLabel = QtWidgets.QLabel(
            "Spatial Sigma: ", self.bilateralFilterGroupBox)
        self.bilateralFilterSigmaSLabel.move(10, 55)

        self.bilateralFilterSigmaSValue = QtWidgets.QLineEdit("2",
            self.bilateralFilterGroupBox)
        self.bilateralFilterSigmaSValue.setGeometry(
            QtCore.QRect(90, 52, 80, 20))

        self.maximumFilterGB = QtWidgets.QGroupBox(
            "Maximum Filter", self.filterGroupBox)
        self.maximumFilterGB.setGeometry(QtCore.QRect(140, 15, 200, 110))
        self.maximumFilterGB.setObjectName("maximumFilterGB")
        self.maximumFilterGB.setVisible(False)

        self.mximumKernelSizeLabel = QtWidgets.QLabel(
            "Kernel size: ", self.maximumFilterGB)
        self.mximumKernelSizeLabel.move(10, 25)

        self.maximumKernelSize = QtWidgets.QLineEdit("5",
            self.maximumFilterGB)
        self.maximumKernelSize.setGeometry(QtCore.QRect(80, 22, 80, 20))

        self.filterButton = QtWidgets.QPushButton(
            "Filter", self.filterGroupBox)
        self.filterButton.setGeometry(QtCore.QRect(10, 100, 120, 25))
        self.filterButton.setObjectName("filterButton")
        self.filterButton.clicked.connect(self.denoise)

        ## Segmentation Groub Box ##

        self.segmentationGroupBox = QtWidgets.QGroupBox("Segmentation", self)
        self.segmentationGroupBox.setGeometry(
            QtCore.QRect(245, 265, 470, 170))
        self.segmentationGroupBox.setObjectName("segmentationGroupBox")
        self.segmentationGroupBox.setEnabled(False)

        self.segmentationList = QtWidgets.QListWidget(
            self.segmentationGroupBox)
        self.segmentationList.setGeometry(QtCore.QRect(10, 20, 125, 110))
        self.segmentationList.setObjectName("segmentationList")
        self.segmentationList.addItem('Manual Threshold')
        self.segmentationList.addItem('Automatic Thresh')
        self.segmentationList.addItem('Edge Operators')
        self.segmentationList.addItem('Canny Edge Detector')
        self.segmentationList.itemDoubleClicked.connect(
            self.segmentationPosition)

        self.segmentButton = QtWidgets.QPushButton(
            "Segmentation", self.segmentationGroupBox)
        self.segmentButton.setGeometry(QtCore.QRect(10, 135, 125, 25))
        self.segmentButton.setObjectName("segmentButton")
        self.segmentButton.clicked.connect(self.segmentation)
        self.segmentButton.setEnabled(False)

        self.thresholdManGB = QtWidgets.QGroupBox(
            "Manual Threshold", self.segmentationGroupBox)
        self.thresholdManGB.setGeometry(QtCore.QRect(145, 15, 300, 110))
        self.thresholdManGB.setObjectName("thresholdManGB")
        self.thresholdManGB.setVisible(False)

        self.thresholdManSlider = QtWidgets.QSlider(self.thresholdManGB)
        self.thresholdManSlider.setOrientation(QtCore.Qt.Horizontal)
        self.thresholdManSlider.setObjectName("thresholdManSlider")
        self.thresholdManSlider.setGeometry(QtCore.QRect(10, 50, 245, 20))
        self.thresholdManSlider.sliderReleased.connect(self.segmentation)

        self.thresholdManValue = QtWidgets.QLineEdit(self.thresholdManGB)
        self.thresholdManValue.setGeometry(QtCore.QRect(260, 50, 35, 20))
        self.thresholdManValue.setEnabled(False)

        self.otsuThreshGB = QtWidgets.QGroupBox(
            "Automatic Threshold", self.segmentationGroupBox)
        self.otsuThreshGB.setGeometry(QtCore.QRect(145, 15, 200, 110))
        self.otsuThreshGB.setObjectName("otsuThreshGB")
        self.otsuThreshGB.setVisible(False)

        self.autoThresh = QtWidgets.QComboBox(
            self.otsuThreshGB)
        self.autoThresh.setGeometry(
            QtCore.QRect(20, 20, 80, 20))
        self.autoThresh.addItem("Otsu")
        self.autoThresh.addItem("Iso")
        self.autoThresh.addItem("Triangle")

        self.swSegmentationGB = QtWidgets.QGroupBox(
            "Edge Operators", self.segmentationGroupBox)
        self.swSegmentationGB.setGeometry(QtCore.QRect(145, 15, 200, 115))
        self.swSegmentationGB.setObjectName("swSegmentationGB")
        self.swSegmentationGB.setVisible(False)

        self.edgesList = QtWidgets.QComboBox(
            self.swSegmentationGB)
        self.edgesList.setGeometry(
            QtCore.QRect(20, 20, 80, 20))
        self.edgesList.addItem("Sobel")
        self.edgesList.addItem("Laplace")
        self.edgesList.addItem("Prewitt")
        self.edgesList.addItem("Roberts")

        self.edgesButton = QtWidgets.QPushButton("Create edges",self.swSegmentationGB)
        self.edgesButton.setGeometry(
            QtCore.QRect(20, 50, 80, 20))
        self.edgesButton.clicked.connect(self.edges)

        self.cannyEdgeDetecetorGB = QtWidgets.QGroupBox(
            "Canny Edge Detector", self.segmentationGroupBox)
        self.cannyEdgeDetecetorGB.setGeometry(QtCore.QRect(145, 15, 200, 115))
        self.cannyEdgeDetecetorGB.setObjectName("cannyEdgeDetecetorGB")
        self.cannyEdgeDetecetorGB.setVisible(False)

        self.horizontCannyLayoutWidget = QtWidgets.QWidget(
            self.cannyEdgeDetecetorGB)
        self.horizontCannyLayoutWidget.setGeometry(
            QtCore.QRect(10, 20, 120, 25))
        self.horizontCannyLayoutWidget.setObjectName(
            "horizontalLayoutWidget_2")

        self.horizontCannyLayout = QtWidgets.QHBoxLayout(
            self.horizontCannyLayoutWidget)
        self.horizontCannyLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontCannyLayout.setObjectName(
            "horizontCannyLayout")

        self.sigmaCannyValueLabel = QtWidgets.QLabel(
            "Sigma Value: ", self.horizontCannyLayoutWidget)
        self.sigmaCannyValueLabel.move(10, 10)
        self.horizontCannyLayout.addWidget(self.sigmaCannyValueLabel)

        self.sigmaCannyValue = QtWidgets.QLineEdit(
            "1", self.horizontCannyLayoutWidget)
        self.sigmaCannyValue.setGeometry(QtCore.QRect(20, 10, 5, 5))
        self.horizontCannyLayout.addWidget(self.sigmaCannyValue)

        self.cannyThreshLayoutWidget = QtWidgets.QWidget(
            self.cannyEdgeDetecetorGB)
        self.cannyThreshLayoutWidget.setGeometry(QtCore.QRect(10, 45, 180, 60))
        self.cannyThreshLayoutWidget.setObjectName("cannyThreshLayoutWidget")

        self.cannyThreshLayout = QtWidgets.QVBoxLayout(
            self.cannyThreshLayoutWidget)
        self.cannyThreshLayout.setContentsMargins(0, 0, 0, 0)
        self.cannyThreshLayout.setObjectName("cannyThreshLayout")

        self.lowThresholdLabel = QtWidgets.QLabel(
            "Low threshold:", self.cannyThreshLayoutWidget)
        self.cannyThreshLayout.addWidget(self.lowThresholdLabel)

        self.lowThresholdSlider = QtWidgets.QSlider(self.cannyThreshLayoutWidget)
        self.lowThresholdSlider.setOrientation(QtCore.Qt.Horizontal)
        self.lowThresholdSlider.setObjectName("lowThresholdSlider")
        self.cannyThreshLayout.addWidget(self.lowThresholdSlider)
        self.lowThresholdSlider.sliderReleased.connect(self.canny)

        self.highThresholdLabel = QtWidgets.QLabel(
            "High threshold:", self.cannyThreshLayoutWidget)
        self.cannyThreshLayout.addWidget(self.highThresholdLabel)

        self.highThresholdSlider = QtWidgets.QSlider(
            self.cannyThreshLayoutWidget)
        self.highThresholdSlider.setOrientation(QtCore.Qt.Horizontal)
        self.highThresholdSlider.setObjectName("highThresholdSlider")
        self.cannyThreshLayout.addWidget(self.highThresholdSlider)
        self.highThresholdSlider.sliderReleased.connect(self.canny)

        self.trackingGroupBox = QtWidgets.QGroupBox("Tracking", self)
        self.trackingGroupBox.setGeometry(
            QtCore.QRect(245, 440, 470, 55))
        self.trackingGroupBox.setObjectName("trackingGroupBox")
        self.trackingGroupBox.setEnabled(False)

        self.trackingHorizontWidget = QtWidgets.QWidget(
            self.trackingGroupBox)
        self.trackingHorizontWidget.setGeometry(
            QtCore.QRect(10, 20, 300, 30))
        self.trackingHorizontWidget.setObjectName(
            "horizontalLayoutWidget_2")

        self.trackingHorizontLayout = QtWidgets.QHBoxLayout(
            self.trackingHorizontWidget)
        self.trackingHorizontLayout.setContentsMargins(0, 0, 0, 0)
        self.trackingHorizontLayout.setObjectName(
            "horizontCannyLayout")
        self.trackingGroupBox.setEnabled(False)

        self.labelsButton = QtWidgets.QPushButton(
            "Labels", self.trackingHorizontWidget)
        self.trackingHorizontLayout.addWidget(self.labelsButton)
        self.labelsButton.clicked.connect(self.labelling)

        self.neighborhoodWindowLabel = QtWidgets.QLabel("Enlarge window by:", self.trackingGroupBox)
        self.neighborhoodWindowLabel.move(60, 10)

        self.neighborhoodWindow = QtWidgets.QLineEdit("5",self.trackingHorizontWidget)
        self.trackingHorizontLayout.addWidget(self.neighborhoodWindow)

        self.neighborhoodX = QtWidgets.QLabel("x", self.trackingHorizontWidget)
        self.trackingHorizontLayout.addWidget(self.neighborhoodX)

        self.neighborhoodWindow2 = QtWidgets.QLineEdit("5", self.trackingHorizontWidget)
        self.trackingHorizontLayout.addWidget(self.neighborhoodWindow2)

        self.startTrackingButton = QtWidgets.QPushButton(
            "Start", self.trackingHorizontWidget)
        self.trackingHorizontLayout.addWidget(self.startTrackingButton)
        self.startTrackingButton.clicked.connect(self.startTracking)
        self.startTrackingButton.setEnabled(False)

        self.stopTrackingButton = QtWidgets.QPushButton(
            "Stop", self.trackingHorizontWidget)
        self.trackingHorizontLayout.addWidget(self.stopTrackingButton)
        self.stopTrackingButton.clicked.connect(self.stopTracking)
        self.stopTrackingButton.setEnabled(False)

        self.parametersWindowLabel = QtWidgets.QLabel("Min size objects:", self.trackingGroupBox)
        self.parametersWindowLabel.move(268, 10)

        self.parametersWindow = QtWidgets.QLineEdit("100",self.trackingHorizontWidget)
        self.trackingHorizontLayout.addWidget(self.parametersWindow)

        self.menuBar = self.menuBar()
        self.fileMenu = self.menuBar.addMenu('File')
        self.histogramMenu = self.menuBar.addMenu('Histogram')

        self.openMenuButton = QtWidgets.QAction('Open', self)
        self.openMenuButton.triggered.connect(self.openImages)

        self.saveMenuButton = QtWidgets.QAction('Save image as', self)
        self.saveMenuButton.triggered.connect(self.saveImage)
        self.saveMenuButton.setEnabled(False)

        self.closeMenuButton = QtWidgets.QAction('Close', self)
        self.closeMenuButton.triggered.connect(self.closeApp)

        self.histogramMenuButton = QtWidgets.QAction('Histogram', self)
        self.histogramMenuButton.triggered.connect(self.histogram)
        self.histogramMenuButton.setEnabled(False)


        self.histogramMenu.addAction(self.histogramMenuButton)
        self.fileMenu.addAction(self.openMenuButton)
        self.fileMenu.addAction(self.saveMenuButton)
        self.fileMenu.addAction(self.closeMenuButton)


        self.errorWarning = QtWidgets.QMessageBox()
        self.errorWarning.setWindowTitle("Invalid Value")

    #method called after clicking on "Open" option in menu
    #it opens selected images from a direcotory, displays a list of selected images
    #and calls method to display the first image in the list on screen
    #also enables adittional button
    def openImages(self):
        # importing paths of images
        self.images = []
        imagePaths, _ = QFileDialog.getOpenFileNames(
            self, 'Open Images', "C:", '*.tif')

        # enabling some widgets of user interface
        self.contrastGroupBox.setEnabled(True)
        self.pathsList.setEnabled(True)
        self.backgroundGroupBox.setEnabled(True)

        # creating objects of imported images
        for path in imagePaths:
            self.images.append(Image(path))

        # filling the list widget with imported paths
        self.pathsList.clear()
        for image in self.images:
            self.pathsList.addItem(image.name)

        # displaying the first image in the list
        if len(self.images) > 0:
            self.index = 0
            self.currentImage = self.images[self.index].loadImage()
            self.positionLabel.setText(self.images[self.index].name)
            self.showImg(self.currentImage)
            self.saveMenuButton.setEnabled(True)
            self.histogramMenuButton.setEnabled(True)
            self.objectPropertiesTable.clearContents()

        # in case we would already do some changes to image and open
        # new file of images, we want to disable some of the funcionality
        self.dilationButton.setEnabled(False)
        self.meanBackgroundButton.setEnabled(False)
        self.backgroundSlider.setEnabled(False)
        self.filterGroupBox.setEnabled(False)
        self.segmentButton.setEnabled(False)
        self.segmentationGroupBox.setEnabled(False)
        self.objectPropertiesTable.setEnabled(False)
        self.startTrackingButton.setEnabled(False)
        self.trackingGroupBox.setEnabled(False)
        self.tableButtonsGroupBox.setEnabled(False)
        self.currentState = "Contrast"

    #method used for displaying image on screen
    def showImg(self, img):
        # checking for color image
        if len(img.shape) > 2:
            # creating a pixmap from color image
            self.pixmap = QPixmap.fromImage(array2qimage(
                img, normalize=(img.min(), img.max()))).scaledToHeight(870)
        else:
            # creating the pixmap from grayscale image
            self.pixmap = QPixmap.fromImage(gray2qimage(
                img, normalize=(img.min(), img.max()))).scaledToHeight(870)

        # displaying the pixmap
        self.imageWindow.setPixmap(self.pixmap)

    #create and show histogram of current image on screen
    def histogram(self):
        hist, centers = skimage.exposure.histogram(self.currentImage, nbins = 4096)
        # cumulative hist
        img, bins = skimage.exposure.cumulative_distribution(self.currentImage, nbins = 4096)
        fig, ax = plt.subplots(1,1, figsize = (8,4))

        ax.plot(centers, hist)
        ax.set_xlabel('jas')
        ax.set_ylabel('četnost')

        ax2 = ax.twinx()
        ax2.plot (bins, img, 'r')
        ax2.set_ylabel('Relativní četnost')

        plt.show()

    #method for manipulation of image contrast - only original data can be manipulated
    def contrast(self, index = None):
        # taking values from user interface

        if index != None:
            ind = index
        else:
            ind = self.index
        img = self.images[ind]
        # if we have displayed filtered image we change the sliders
        if self.currentState == "Mean Background":
            self.blackSlider.setMaximum(1000000)
            self.whiteSlider.setMaximum(1000000)
            valueBlack = self.blackSlider.value() /1000000
            valueWhite = self.whiteSlider.value() /1000000
            image = img.contrast(valueBlack, valueWhite, self.currentImage)
        else:
            self.blackSlider.setMaximum(4096)
            self.whiteSlider.setMaximum(4096)
            valueBlack = self.blackSlider.value()
            valueWhite = self.whiteSlider.value()
            self.currentState = "Contrast"

            # changing the contrast of the image and showing the result

            image = img.contrast(valueBlack, valueWhite)

        self.showImg(image)

    # method called when user want to change image using the list of opened images
    # it will apply all the methods there have been applied so far
    def pathPosition(self):
        index = self.pathsList.currentRow()

        # applying methods depending on the currently visible image
        if self.currentState == "Contrast":
            self.images[index].loadImage()
            self.contrast(index)

        elif self.currentState == "Median Background":
            neighborhood = self.images[index - (self.neighborhoodSize + 3):index - 3
                            ] + self.images[index + 4:index + (self.neighborhoodSize + 4)]
            self.images[index].loadImage()
            self.showImg(self.images[index].medianBackground(neighborhood))
            self.dilationButton.setEnabled(False)
            self.meanBackgroundButton.setEnabled(False)
            self.filterGroupBox.setEnabled(False)
            self.segmentButton.setEnabled(False)
            self.segmentationGroupBox.setEnabled(False)
            self.objectPropertiesTable.setEnabled(False)
            self.startTrackingButton.setEnabled(False)
            self.trackingGroupBox.setEnabled(False)
            self.tableButtonsGroupBox.setEnabled(False)

        elif self.currentState == "Threshold Background":
            neighborhood = self.images[index - (self.neighborhoodSize + 3):index - 3
                            ] + self.images[index + 4:index + (self.neighborhoodSize + 4)]
            self.images[index].loadImage()
            self.images[index].medianBackground(neighborhood)
            self.showImg(self.images[index].threshold(self.threshold))

            self.meanBackgroundButton.setEnabled(False)
            self.filterGroupBox.setEnabled(False)
            self.segmentButton.setEnabled(False)
            self.segmentationGroupBox.setEnabled(False)
            self.objectPropertiesTable.setEnabled(False)
            self.startTrackingButton.setEnabled(False)
            self.trackingGroupBox.setEnabled(False)
            self.tableButtonsGroupBox.setEnabled(False)

        elif self.currentState == "Dilation":
            neighborhood = self.images[index - (self.neighborhoodSize + 3):index - 3
                            ] + self.images[index + 4:index + (self.neighborhoodSize + 4)]
            self.images[index].loadImage()
            self.images[index].medianBackground(neighborhood)
            self.images[index].threshold(self.threshold)
            self.showImg(self.images[index].dilation())

            self.filterGroupBox.setEnabled(False)
            self.segmentButton.setEnabled(False)
            self.segmentationGroupBox.setEnabled(False)
            self.objectPropertiesTable.setEnabled(False)
            self.startTrackingButton.setEnabled(False)
            self.trackingGroupBox.setEnabled(False)
            self.tableButtonsGroupBox.setEnabled(False)

        elif self.currentState == "Mean Background":
            neighborhood = self.images[index - (self.neighborhoodSize + 3):index - 3
                            ] + self.images[index + 4:index + (self.neighborhoodSize + 4)]
            self.images[index].loadImage()
            self.images[index].medianBackground(neighborhood)
            self.images[index].threshold(self.threshold)
            self.images[index].dilation()
            self.images[index].meanBackground(neighborhood)
            self.showImg(self.images[index].imageDivision())

            self.segmentButton.setEnabled(False)
            self.segmentationGroupBox.setEnabled(False)
            self.objectPropertiesTable.setEnabled(False)
            self.startTrackingButton.setEnabled(False)
            self.trackingGroupBox.setEnabled(False)
            self.tableButtonsGroupBox.setEnabled(False)

        elif self.currentState == "Filtered":
            neighborhood = self.images[index - (self.neighborhoodSize + 3):index - 3
                            ] + self.images[index + 4:index + (self.neighborhoodSize + 4)]
            self.images[index].loadImage()
            self.images[index].medianBackground(neighborhood)
            self.images[index].threshold(self.threshold)
            self.images[index].dilation()
            self.images[index].meanBackground(neighborhood)
            self.images[index].imageDivision()

            self.objectPropertiesTable.setEnabled(False)
            self.startTrackingButton.setEnabled(False)
            self.trackingGroupBox.setEnabled(False)
            self.tableButtonsGroupBox.setEnabled(False)

            self.currentFiltImg = np.zeros_like(self.images[index].data)

            if self.filter == "Median Filter":
                self.currentFiltImg = medianFilter.process(self.images[index].divisionImg, self.kernel)

            elif self.filter == "Mean Filter":
                self.currentFiltImg = meanFilter.process(self.images[index].divisionImg, self.kernel)

            elif self.filter == "Gauss Filter":
                self.currentFiltImg = gaussFilter.process(self.images[index].divisionImg, self.sigma)

            elif self.filter == "Bilateral Filter":
                self.currentFiltImg = bilateralFilter.process(
                    self.images[index].divisionImg, self.sigmaColor, self.sigmaSpatial)

            elif self.filter == "Maximum Filter":
                self.currentFiltImg = maxFilter.process(self.images[index].divisionImg, self.kernel)

            self.showImg(self.currentFiltImg)
        elif self.currentState == "Canny":
            neighborhood = self.images[index - (self.neighborhoodSize + 3):index - 3
                            ] + self.images[index + 4:index + (self.neighborhoodSize + 4)]
            self.images[index].loadImage()
            self.images[index].medianBackground(neighborhood)
            self.images[index].threshold(self.threshold)
            self.images[index].dilation()
            self.images[index].meanBackground(neighborhood)
            self.images[index].imageDivision()


            self.startTrackingButton.setEnabled(False)
            self.tableButtonsGroupBox.setEnabled(False)

            self.currentFiltImg = np.zeros_like(self.images[index].data)

            if self.filter == "Median Filter":
                self.currentFiltImg = medianFilter.process(self.images[index].divisionImg, self.kernel)
                #self.currentFiltImg= np.where(self.images[index].mask == 1, self.currentFiltImg, 1)

            elif self.filter == "Mean Filter":
                self.currentFiltImg = meanFilter.process(self.images[index].divisionImg, self.kernel)

            elif self.filter == "Gauss Filter":
                self.currentFiltImg = gaussFilter.process(self.images[index].divisionImg, self.sigma)

            elif self.filter == "Bilateral Filter":
                self.currentFiltImg = bilateralFilter.process(
                    self.images[index].divisionImg, self.sigmaColor, self.sigmaSpatial)

            elif self.filter == "Maximum Filter":
                self.currentFiltImg = maxFilter.process(self.images[index].divisionImg, self.kernel)

            self.showImg(self.images[index].cannyEdgeDetector(
            self.currentFiltImg, self.sigmaCanny, self.lowThresh, self.highThresh))

        elif self.currentState == "Edges":
            neighborhood = self.images[index - (self.neighborhoodSize + 3):index - 3
                            ] + self.images[index + 4:index + (self.neighborhoodSize + 4)]
            self.images[index].loadImage()
            self.images[index].medianBackground(neighborhood)
            self.images[index].threshold(self.threshold)
            self.images[index].dilation()
            self.images[index].meanBackground(neighborhood)
            self.images[index].imageDivision()

            self.startTrackingButton.setEnabled(False)
            self.tableButtonsGroupBox.setEnabled(False)

            self.currentFiltImg = np.zeros_like(self.images[index].data)

            if self.filter == "Median Filter":
                self.currentFiltImg = medianFilter.process(self.images[index].divisionImg, self.kernel)
                #self.currentFiltImg= np.where(self.images[index].mask == 1, self.currentFiltImg, 1)

            elif self.filter == "Mean Filter":
                self.currentFiltImg = meanFilter.process(self.images[index].divisionImg, self.kernel)

            elif self.filter == "Gauss Filter":
                self.currentFiltImg = gaussFilter.process(self.images[index].divisionImg, self.sigma)

            elif self.filter == "Bilateral Filter":
                self.currentFiltImg = bilateralFilter.process(
                    self.images[index].divisionImg, self.sigmaColor, self.sigmaSpatial)
            elif self.filter == "Maximum Filter":
                self.currentFiltImg = maxFilter.process(self.images[index].divisionImg, self.kernel)

            self.showImg(self.images[index].edges(self.currentFiltImg))

        elif self.currentState == "Segmented":
            neighborhood = self.images[index - (self.neighborhoodSize + 3):index - 3
                            ] + self.images[index + 4:index + (self.neighborhoodSize + 4)]
            self.images[index].loadImage()
            self.images[index].medianBackground(neighborhood)
            self.images[index].threshold(self.threshold)
            self.images[index].dilation()
            self.images[index].meanBackground(neighborhood)
            self.images[index].imageDivision()

            self.startTrackingButton.setEnabled(False)
            self.tableButtonsGroupBox.setEnabled(False)

            self.currentFiltImg = np.zeros_like(self.images[index].data)

            if self.filter == "Median Filter":
                self.currentFiltImg = medianFilter.process(self.images[index].divisionImg, self.kernel)
                #self.currentFiltImg= np.where(self.images[index].mask == 1, self.currentFiltImg, 1)

            elif self.filter == "Mean Filter":
                self.currentFiltImg = meanFilter.process(self.images[index].divisionImg, self.kernel)

            elif self.filter == "Gauss Filter":
                self.currentFiltImg = gaussFilter.process(self.images[index].divisionImg, self.sigma)

            elif self.filter == "Bilateral Filter":
                self.currentFiltImg = bilateralFilter.process(
                    self.images[index].divisionImg, self.sigmaColor, self.sigmaSpatial)

            elif self.filter == "Maximum Filter":
                self.currentFiltImg = maxFilter.process(self.images[index].divisionImg, self.kernel)


            if self.segmentationTech == "Manual Threshold":
                self.showImg(self.images[index].thresholdSegment(self.currentFiltImg, self.manThreshold))

            elif self.segmentationTech == "Automatic Thresh":
                method = self.autoThresh.currentText()
                self.showImg(self.images[index].autoThresh(method, self.currentFiltImg))

            elif self.segmentationTech == "Edge Operators":
                self.images[index].edges(self.currentFiltImg)
                self.showImg(self.images[index].edgesFilling())

            elif self.segmentationTech == "Canny Edge Detector":

                self.images[index].cannyEdgeDetector(self.currentFiltImg, self.sigmaCanny,
                                      self.lowThresh, self.highThresh)
                self.showImg(self.images[index].fillingHoles())

        elif self.currentState == "Labeled":
            neighborhood = self.images[index - (self.neighborhoodSize + 3):index - 3
                            ] + self.images[index + 4:index + (self.neighborhoodSize + 4)]
            self.images[index].loadImage()
            self.images[index].medianBackground(neighborhood)
            self.images[index].threshold(self.threshold)
            self.images[index].dilation()
            self.images[index].meanBackground(neighborhood)
            self.images[index].imageDivision()

            self.currentFiltImg= np.zeros_like(self.images[index].data)

            if self.filter == "Median Filter":
                self.currentFiltImg = medianFilter.process(self.images[index].divisionImg, self.kernel)
                #self.currentFiltImg = np.where(self.images[index].mask == 1, self.currentFiltImg, 1)

            elif self.filter == "Mean Filter":
                self.currentFiltImg = meanFilter.process(self.images[index].divisionImg, self.kernel)

            elif self.filter == "Gauss Filter":
                self.currentFiltImg = gaussFilter.process(self.images[index].divisionImg, self.sigma)

            elif self.filter == "Bilateral Filter":
                self.currentFiltImg = bilateralFilter.process(
                    self.images[index].divisionImg, self.sigmaColor, self.sigmaSpatial)

            elif self.filter == "Maximum Filter":
                self.currentFiltImg = maxFilter.process(self.images[index].divisionImg, self.kernel)


            if self.segmentationTech == "Manual Threshold":
                self.images[index].thresholdSegment(self.currentFiltImg, self.manThreshold)

            elif self.segmentationTech == "Automatic Thresh":
                method = self.autoThresh.currentText()
                self.images[index].autoThresh(method, self.currentFiltImg)

            elif self.segmentationTech == "Edge Operators":
                self.images[index].edges(self.currentFiltImg)
                self.showImg(self.images[index].edgesFilling())

            elif self.segmentationTech == "Canny Edge Detector":

                self.images[index].cannyEdgeDetector(self.currentFiltImg, self.sigmaCanny,
                                      self.lowThresh, self.highThresh)
                self.images[index].fillingHoles()

            self.labelling(index)
            # self.showImg(image)

            # self.objectPropertiesTable.clearContents()
            # self.objectPropertiesTable.setRowCount(1)

            # self.tableUpdate(self.objectProperties)
            # Tracking.zeroCross = []

        self.index = self.pathsList.currentRow()
        self.positionLabel.setText(self.images[self.index].name)

    # method for showing an error message
    def popup(self, icon = 1):
        # choosing the type of warning
        if icon == 1:
            self.errorWarning.setIcon(QtWidgets.QMessageBox.Warning)
        elif icon == 2:
            self.errorWarning.setIcon(QtWidgets.QMessageBox.Critical)
        elif icon == 3:
            self.errorWarning.setIcon(QtWidgets.QMessageBox.Question)
        elif icon == 4:
            self.errorWarning.setIcon(QtWidgets.QMessageBox.Information)

        result = self.errorWarning.exec_()

        return result

    # method which creates a median background
    def backgroundMedian(self):

        # checking for valid parameters
        try:
            value = int(self.neigborhoodSize.text())
            if value <= 0:
                self.errorWarning.setText("The size of neighborhood has to have positive integer value!!")
                self.errorWarning.setWindowTitle("Invalid Value")
                return self.popup()
        except ValueError:
            self.errorWarning.setText("You have to enter positive integer value!")
            self.errorWarning.setWindowTitle("Invalid Value")
            return self.popup()

        img = self.images[self.index]

        # taking parameter from the user interface
        self.neighborhoodSize = int(self.neigborhoodSize.text())

        # checking if we have enough images imported for creation of desired neighborhood
        if (2 * self.neighborhoodSize + 6) >= len(self.images):
            return self.lineEditError.showMessage(f'You need to import at least {2 * (self.neighborhoodSize) + 6} images!')

        # creation of neighborhood by importing all the images
        neighborhood = self.images[self.index - (self.neighborhoodSize + 3):self.index - 3
                                ] + self.images[self.index + 4:self.index + (self.neighborhoodSize + 4)]

        # creating the median background and showing the result
        self.subtractImage = img.medianBackground(neighborhood)
        self.currentImage = self.subtractImage
        self.showImg(self.currentImage)

        # setting a slider for further usage
        maxDec= np.abs(int(math.log10(self.currentImage.max())))
        minDec= np.abs(int(math.log10(self.currentImage.min())))

        maximum = int(np.multiply(np.round(self.currentImage.max(), decimals= minDec+2), 10**(minDec+2)))
        minimum = int(np.multiply(np.round(self.currentImage.min(), decimals= minDec+2), 10**(minDec+2)))

        self.backgroundSlider.setMaximum(maximum)
        self.backgroundSlider.setMinimum(minimum)
        self.backgroundSlider.setSingleStep(1)
        self.backgroundSlider.setEnabled(True)

        self.currentState = "Median Background"

    def thresholdBackground(self):
        img = self.images[self.index]
        minDec = np.abs(int(math.log10(self.subtractImage.min())))

        # reading a threshold from slider in the UI
        self.threshold = np.divide(self.backgroundSlider.value(),10**(minDec+2))
        self.thresholdWindow.setText(str(self.threshold))

        self.dilationButton.setEnabled(True)

        # creat a thresholded image and displaying the result
        self.currentImage = img.threshold(self.threshold)
        self.currentState = "Threshold Background"
        self.showImg(self.currentImage)

    def dilation(self):

        img = self.images[self.index]

        # applying dilation to the thresholded image and displaying result
        self.meanBackgroundButton.setEnabled(True)
        self.currentImage = img.dilation()
        self.currentState = "Dilation"
        self.showImg(self.currentImage)

    def backgroundMean(self):

        img = self.images[self.index]

        # checking for valid parameters
        try:
            value = int(self.neigborhoodSize.text())
            if value <= 0:
                self.errorWarning.setText("The size of neighborhood has to have positive integer value!!")
                self.errorWarning.setWindowTitle("Invalid Value")
                return self.popup()
        except ValueError:
            self.errorWarning.setText("You have to enter positive integer value!")
            self.errorWarning.setWindowTitle("Invalid Value")
            return self.popup()

        neighborhoodSize = int(self.neigborhoodSize.text())

        # checking if we have enough images imported for creation of desired neighborhood
        if (2 * self.neighborhoodSize + 6) >= len(self.images):
            return self.lineEditError.showMessage(f'You need to import at least {2 * (self.neighborhoodSize) + 6} images!')

        # creation of neighborhood by importing all the images
        neighborhood = self.images[self.index - (neighborhoodSize + 3):self.index - 3
                                   ] + self.images[self.index + 4:self.index + (neighborhoodSize + 4)]

        # creating the mean background and calling next method
        img.meanBackground(neighborhood)
        self.imageDivision()

    def imageDivision(self):
        img = self.images[self.index]

        # division of original image by mean background and displaying result
        self.currentImage = img.imageDivision()
        self.currentState = "Mean Background"
        self.showImg(self.currentImage)

        # enabeling additional widgets in UI
        self.filterGroupBox.setEnabled(True)
        self.medianFilterGroupBox.setVisible(True)

    # checking selected filtration method
    def filterPosition(self):
        self.filter = self.filterList.currentItem().text()

        # disabling all the boxes for different filtration methods
        self.bilateralFilterGroupBox.setVisible(False)
        self.medianFilterGroupBox.setVisible(False)
        self.gaussFilterGroupBox.setVisible(False)
        self.meanFilterGroupBox.setVisible(False)
        self.maximumFilterGB.setVisible(False)

        # enabeling box with selected filtration method
        if self.filter == "Median Filter":
            self.medianFilterGroupBox.setVisible(True)

        elif self.filter == "Mean Filter":
            self.meanFilterGroupBox.setVisible(True)

        elif self.filter == "Gauss Filter":
            self.gaussFilterGroupBox.setVisible(True)

        elif self.filter == "Bilateral Filter":
            self.bilateralFilterGroupBox.setVisible(True)

        elif self.filter == "Maximum Filter":
            self.maximumFilterGB.setVisible(True)
    # applying specific image filtration method
    def denoise(self):
        image = self.images[self.index]

        # enabeling segmentation box for further analysis
        self.segmentationGroupBox.setEnabled(True)
        self.currentState = "Filtered"

        # applying specific filtration method on an image
        if self.filter == "Median Filter":
            # checking if the parameters are suitable
            try:
                value = int(self.medianFilterKernelSize.text())
                if value < 0:
                    self.errorWarning.setText("The size of kernel has to have positive and odd integer value!!")
                    self.errorWarning.setWindowTitle("Invalid Value")
                    return self.popup()
            except ValueError:
                self.errorWarning.setText("You have to enter positive and odd integer value!")
                self.errorWarning.setWindowTitle("Invalid Value")
                return self.popup()

            try:
                value = int(self.medianFilterKernelSize.text())
                if value%2 == 0:
                    self.errorWarning.setText("The size of kernel has to have positive and odd integer value!!")
                    self.errorWarning.setWindowTitle("Invalid Value")
                    return self.popup()
            except ValueError:
                self.errorWarning.setText("You have to enter positive and odd integer value!")
                self.errorWarning.setWindowTitle("Invalid Value")
                return self.popup()

            # taking the parameters from UI
            geometry = str(self.medianFilterKernelGeometry.currentText())
            size = int(self.medianFilterKernelSize.text())

            if geometry == "Square":
                self.kernel = skimage.morphology.square(size)
            else:
                self.kernel = skimage.morphology.disk(size)

            # applying median filter on image using "filter" module and specific static method
            self.currentFiltImg = medianFilter.process(image.divisionImg, self.kernel)
            self.currentImage = self.currentFiltImg

        elif self.filter == "Mean Filter":

            try:
                value = int(self.meanFilterKernelSize.text())
                if value < 0:
                    self.errorWarning.setText("The size of kernel has to have positive and odd integer value!!")
                    self.errorWarning.setWindowTitle("Invalid Value")
                    return self.popup()
            except ValueError:
                self.errorWarning.setText("You have to enter positive and odd integer value!")
                self.errorWarning.setWindowTitle("Invalid Value")
                return self.popup()

            try:
                value = int(self.meanFilterKernelSize.text())
                if value % 2 == 0:
                    self.errorWarning.setText("The size of kernel has to have positive and odd integer value!!")
                    self.errorWarning.setWindowTitle("Invalid Value")
                    return self.popup()
            except ValueError:
                self.errorWarning.setText("You have to enter positive and odd integer value!")
                self.errorWarning.setWindowTitle("Invalid Value")
                return self.popup()


            self.sizeKernel = int(self.meanFilterKernelSize.text())

            self.currentFiltImg = meanFilter.process(image.divisionImg, self.sizeKernel)
            self.currentImage = self.currentFiltImg

        elif self.filter == "Gauss Filter":
            # checking if the parameters are suitable
            try:
                value = float(self.gaussFilterSigmaValue.text())
                if value < 0:
                    self.errorWarning.setText("The color has to have positive scalar value!!")
                    self.errorWarning.setWindowTitle("Invalid Value")
                    return self.popup()
            except ValueError:
                self.errorWarning.setText("You have to enter positive scalar value as sigma!")
                self.errorWarning.setWindowTitle("Invalid Value")
                return self.popup()

            # taking the parameters from UI
            self.sigma = float(self.gaussFilterSigmaValue.text())

            # applying gauss filter on image using "filter" module and specific static method
            self.currentFiltImg = gaussFilter.process(image.divisionImg, self.sigma)
            self.currentImage = self.currentFiltImg

        elif self.filter == "Bilateral Filter":
            # checking if the parameters are suitable
            try:
                value = float(self.bilateralFilterSigmaCValue.text())
                if value < 0:
                    self.errorWarning.setText("The sigma color has to have positive scalar value!!")
                    self.errorWarning.setWindowTitle("Invalid Value")
                    return self.popup()
            except ValueError:
                self.errorWarning.setText("You have to enter positive scalar value as sigma color!")
                self.errorWarning.setWindowTitle("Invalid Value")
                return self.popup()

            try:
                value = float(self.bilateralFilterSigmaSValue.text())
                if value < 0:
                    self.errorWarning.setText("The sigma spatial has to have positive scalar value!!")
                    self.errorWarning.setWindowTitle("Invalid Value")
                    return self.popup()
            except ValueError:
                self.errorWarning.setText("You have to enter positive scalar value as sigma spatial!")
                self.errorWarning.setWindowTitle("Invalid Value")
                return self.popup()

            # taking the parameters from UI
            self.sigmaColor = float(self.bilateralFilterSigmaCValue.text())
            self.sigmaSpatial = float(self.bilateralFilterSigmaSValue.text())

            # applying gauss filter on image using "filter" module and specific static method
            self.currentFiltImg = bilateralFilter.process(image.divisionImg, self.sigmaColor, self.sigmaSpatial)
            self.currentImage = self.currentFiltImg


        elif self.filter == "Maximum Filter":
            try:
                value = int(self.maximumKernelSize.text())
                if value < 0:
                    self.errorWarning.setText("The size of kernel has to have positive and odd integer value!!")
                    self.errorWarning.setWindowTitle("Invalid Value")
                    return self.popup()
            except ValueError:
                self.errorWarning.setText("You have to enter positive and odd integer value!")
                self.errorWarning.setWindowTitle("Invalid Value")
                return self.popup()

            try:
                value = int(self.maximumKernelSize.text())
                if value % 2 == 0:
                    self.errorWarning.setText("The size of kernel has to have positive and odd integer value!!")
                    self.errorWarning.setWindowTitle("Invalid Value")
                    return self.popup()
            except ValueError:
                self.errorWarning.setText("You have to enter positive and odd integer value!")
                self.errorWarning.setWindowTitle("Invalid Value")
                return self.popup()

            # taking the parameters from UI

            self.sizeKernel = int(self.maximumKernelSize.text())

            self.currentFiltImg = maxFilter.process(image.divisionImg, self.sizeKernel)

        self.showImg(self.currentFiltImg)

    # checking the position in segmetation method list
    def segmentationPosition(self):
        self.segmentationTech = self.segmentationList.currentItem().text()

        # disabeling all the boxes of the different methods
        self.thresholdManGB.setVisible(False)
        self.otsuThreshGB.setVisible(False)
        self.swSegmentationGB.setVisible(False)
        self.cannyEdgeDetecetorGB.setVisible(False)

        # enabeling specific box depending on selected segmentation method
        if self.segmentationTech == "Manual Threshold":
            # changing settings of slider
            maximum = np.multiply(
                np.round(self.currentFiltImg.max(), decimals=3), 1000)
            minimum = np.multiply(
                np.round(self.currentFiltImg.min(), decimals=3), 1000)
            tick = int(
                np.round(np.divide(np.subtract(
                    maximum, minimum), 100), decimals=0))

            if tick == 0:
                tick = 1

            self.thresholdManSlider.setMaximum(maximum)
            self.thresholdManSlider.setMinimum(minimum)
            self.thresholdManSlider.setSingleStep(tick)

            self.thresholdManGB.setVisible(True)

        elif self.segmentationTech == "Automatic Thresh":
            self.segmentButton.setEnabled(True)
            self.otsuThreshGB.setVisible(True)

        elif self.segmentationTech == "Edge Operators":
            self.swSegmentationGB.setVisible(True)

        elif self.segmentationTech == "Canny Edge Detector":
            #changing settings of the sliders
            self.lowThresholdSlider.setMaximum(300000)
            self.lowThresholdSlider.setMinimum(0)
            self.lowThresholdSlider.setSingleStep(1)
            self.highThresholdSlider.setMaximum(300000)
            self.highThresholdSlider.setMinimum(0)
            self.highThresholdSlider.setSingleStep(1)
            self.highThresholdSlider.setProperty("value", 100000)
            self.cannyEdgeDetecetorGB.setVisible(True)
            self.segmentButton.setEnabled(True)


    def segmentation(self):
        image = self.images[self.index]

        self.trackingGroupBox.setEnabled(True)
        self.currentState = "Segmented"

        # applying specific segmentation method on filtered image
        if self.segmentationTech == "Manual Threshold":
            # taking parameters from UI
            self.manThreshold = np.divide(self.thresholdManSlider.value(), 1000)
            self.thresholdManValue.setText(str(self.manThreshold))

            # apply the threshold on filtered image and displazing the result
            self.currentImage = image.thresholdSegment(
                self.currentFiltImg, self.manThreshold)

            self.showImg(self.currentImage)

        elif self.segmentationTech == "Automatic Thresh":
            # take selected auto threshold method
            self.autoMethod = self.autoThresh.currentText()
            # apply this method to the filtered image and display the result
            self.currentImage = image.autoThresh(self.autoMethod, self.currentFiltImg)
            self.showImg(self.currentImage)

        elif self.segmentationTech == "Edge Operators":
            # filling the edges created by one of the edge finding methods
            self.currentImage = image.edgesFilling()
            self.showImg(self.currentImage)

        elif self.segmentationTech == "Canny Edge Detector":
            # filling the edges found by CED
            self.currentImage= image.fillingHoles()
            self.showImg(self.currentImage)

    def edges(self):
        image = self.images[self.index]
        self.segmentButton.setEnabled(True)

        # taking the selected method for finding edges
        self.autoMethod = self.edgesList.currentText()

        # applying this method on filtered image and create an edge map
        self.currentImage = image.edges(self.autoMethod, self.currentFiltImg)
        self.showImg(self.currentImage)
        self.currentState = 'Edges'

    def canny(self):
        image = self.images[self.index]

        # check if the parameters from UI are suitable
        try:
            value = float(self.sigmaCannyValue.text())
            if value < 0:
                self.errorWarning.setText("The sigma has to have positive scalar value!!")
                self.errorWarning.setWindowTitle("Invalid Value")
                return self.popup()
        except ValueError:
            self.errorWarning.setText("You have to enter positive scalar value as sigma !")
            self.errorWarning.setWindowTitle("Invalid Value")
            return self.popup()

        # take the parameters from UI
        self.sigmaCanny = float(self.sigmaCannyValue.text())
        minDec= np.abs(int(math.log10(self.currentFiltImg.min())))

        # change the values
        self.lowThresh = np.divide(self.lowThresholdSlider.value(),10**7)
        self.highThresh = np.divide(self.highThresholdSlider.value(),10**7)

        # create an edge map using Canny Edge Detector
        self.currentState = "Canny"
        self.currentImage = image.cannyEdgeDetector(
            self.currentFiltImg, self.sigmaCanny, self.lowThresh, self.highThresh)
        self.showImg(self.currentImage)


    def labelling(self, index = None):
        if index != False:
            ind = index
        else:
            ind = self.index


        image = self.images[ind]
        self.currentState = "Labeled"

        # create list for parameters

        self.uniqueLabel = []
        self.labelCount = [0] * 100
        self.pozX = [[] for x in range(100)]
        self.pozY = [[] for x in range(100)]
        self.orientation = [[] for x in range(100)]
        self.changedLabels = {}
        self.ellongation = [[] for x in range(100)]

        # enable properties table and tracking button
        self.objectPropertiesTable.setEnabled(True)
        self.objectPropertiesTable.removeRow(0)
        self.tableButtonsGroupBox.setEnabled(True)
        self.objectPropertiesTable.setColumnCount(len(self.labels))
        self.objectPropertiesTable.setHorizontalHeaderLabels(self.labels)
        self.objectPropertiesTable.setEditTriggers(QtWidgets.QTableWidget.NoEditTriggers)

        self.startTrackingButton.setEnabled(True)

        # get properties of objects in image and color image
        try:
            value = int(self.parametersWindow.text())
            if value < 20 :
                self.errorWarning.setText("The size has to have integer value larger than 19!!")
                self.errorWarning.setWindowTitle("Invalid Value")
                return self.popup()
        except ValueError:
            self.errorWarning.setText("You have to enter integer value larger than 19!")
            self.errorWarning.setWindowTitle("Invalid Value")
            return self.popup()
        self.size = int(self.parametersWindow.text())
        self.objectProperties, _ , self.currentImage = image.labelling(self.size)

        self.objectPropertiesTable.clearContents()
        self.objectPropertiesTable.setRowCount(1)

        Tracking.zeroCross = []
        Tracking.sameLabelAlive = None
        Tracking.propertiesNew = None

        # update table of properties and display image
        self.tableUpdate(self.objectProperties)
        self.showImg(self.currentImage)

    def tableUpdate(self, objectProperties, orientation = None, ellongation = None):
        # counting all the rows
        rowCount = self.objectPropertiesTable.rowCount()
        if rowCount == 1 and self.objectPropertiesTable.item(0,0) == None:
            self.objectPropertiesTable.removeRow(0)
        rowCount = self.objectPropertiesTable.rowCount()
        self.objectProperties = objectProperties
        # creating list of data
        datas = ['']*len(self.labels)

        # iteration over all the obects in given objectProperties
        for i, (fiber,color) in enumerate(zip(objectProperties, self.colors)):
            label = None
            # create a new row in the table
            self.objectPropertiesTable.insertRow(rowCount)
            # get centroid position of the current object
            centroidx, centroidy = fiber.centroid
            if len(self.pozX[fiber.label]) > 0:
                if self.pozX[fiber.label][-1] - centroidy > 100 or self.pozY[fiber.label][-1] - centroidx >50:
                    if fiber.label in self.changedLabels:
                        label = self.changedLabels[fiber.label]
                        self.labelCount[label] += 1
                        # update position list of specific object
                        self.pozX[label].append(centroidy)
                        self.pozY[label].append(centroidx)
                        if orientation:
                            self.orientation[label].append(orientation[i])
                        else:
                            self.orientation[label].append(np.round(math.pi/2 + fiber.orientation,decimals = 2))
                        if ellongation:
                            self.ellongation[label].append(ellongation[i])
                        else:
                            majorA = fiber.major_axis_length
                            minorA = fiber.minor_axis_length
                            self.ellongation[label].append(math.log2(majorA/minorA))

                    else:
                        label = max(self.uniqueLabel) + 1
                        self.uniqueLabel.append(label)
                        self.changedLabels[fiber.label] = label
                        self.labelCount[label] += 1
                        # update position list of specific object
                        self.pozX[label].append(centroidy)
                        self.pozY[label].append(centroidx)
                        if orientation:
                            self.orientation[label].append(orientation[i])
                        else:
                            self.orientation[label].append(np.round(math.pi/2 + fiber.orientation,decimals = 2))
                        if ellongation:
                            self.ellongation[label].append(ellongation[i])
                        else:
                            majorA = fiber.major_axis_length
                            minorA = fiber.minor_axis_length
                            self.ellongation[label].append(math.log2(majorA/minorA))

                else:
                    if fiber.label in self.uniqueLabel:
                        # count number of records for specific label
                        self.labelCount[fiber.label] += 1
                        # update position list of specific object
                        self.pozX[fiber.label].append(centroidy)
                        self.pozY[fiber.label].append(centroidx)
                        if orientation:
                            self.orientation[fiber.label].append(orientation[i])
                        else:
                            self.orientation[fiber.label].append(np.round(math.pi/2 + fiber.orientation,decimals = 2))
                        if ellongation:
                            self.ellongation[fiber.label].append(ellongation[i])
                        else:
                            majorA = fiber.major_axis_length
                            minorA = fiber.minor_axis_length
                            self.ellongation[fiber.label].append(math.log2(majorA/minorA))
                    else:
                        # add new label to the list of labels in table
                        self.uniqueLabel.append(fiber.label)
                        self.labelCount[fiber.label] += 1
                        self.pozX[fiber.label].append(centroidy)
                        self.pozY[fiber.label].append(centroidx)
                        if orientation:
                            self.orientation[fiber.label].append(orientation[i])
                        else:
                            self.orientation[fiber.label].append(np.round(math.pi/2 + fiber.orientation,decimals = 2))
                        if ellongation:
                            self.ellongation[fiber.label].append(ellongation[i])
                        else:
                            majorA = fiber.major_axis_length
                            minorA = fiber.minor_axis_length
                            self.ellongation[fiber.label].append(math.log2(majorA/minorA))
            else:
                    if fiber.label in self.uniqueLabel:
                        # count number of records for specific label
                        self.labelCount[fiber.label] += 1
                        # update position list of specific object
                        self.pozX[fiber.label].append(centroidy)
                        self.pozY[fiber.label].append(centroidx)
                        if orientation:
                            self.orientation[fiber.label].append(orientation[i])
                        else:
                            self.orientation[fiber.label].append(np.round(math.pi/2 + fiber.orientation,decimals = 2))
                        if ellongation:
                            self.ellongation[fiber.label].append(ellongation[i])
                        else:
                            majorA = fiber.major_axis_length
                            minorA = fiber.minor_axis_length
                            self.ellongation[fiber.label].append(math.log2(majorA/minorA))
                    else:
                        # add new label to the list of labels in table
                        self.uniqueLabel.append(fiber.label)
                        self.labelCount[fiber.label] += 1
                        self.pozX[fiber.label].append(centroidy)
                        self.pozY[fiber.label].append(centroidx)
                        if orientation:
                            self.orientation[fiber.label].append(orientation[i])
                        else:
                            self.orientation[fiber.label].append(np.round(math.pi/2 + fiber.orientation,decimals = 2))
                        if ellongation:
                            self.ellongation[fiber.label].append(ellongation[i])
                        else:
                            majorA = fiber.major_axis_length
                            minorA = fiber.minor_axis_length
                            self.ellongation[fiber.label].append(math.log2(majorA/minorA))

            # rouding the centroid position and creating string
            centroidx = str(np.round(centroidx,decimals = 2))
            centroidy = str(np.round(centroidy,decimals = 2))

            if orientation:
                # create data to put into table
                if fiber.label in self.changedLabels:
                    datas = [str(self.changedLabels[fiber.label]),str(color),centroidy,centroidx,str(fiber.area),str(orientation[i]), str(ellongation[i])]
                else:
                    datas = [str(fiber.label),str(color),centroidy,centroidx,str(fiber.area),str(orientation[i]),str(ellongation[i])]
            else:
                orient = str(np.round(math.pi/2 + fiber.orientation,decimals = 2))
                majorA = fiber.major_axis_length
                minorA = fiber.minor_axis_length
                ellong = np.round(math.log2(majorA / minorA), decimals = 2)

                if fiber.label in self.changedLabels:
                    datas = [str(self.changedLabels[fiber.label]),str(color),centroidy,centroidx,str(fiber.area),orient,str(ellong)]
                else:
                    datas = [str(fiber.label),str(color),centroidy,centroidx,str(fiber.area),orient, str(ellong)]

            # add the data into the table
            for j, data in enumerate(datas):
                self.objectPropertiesTable.setItem(rowCount, j, QtWidgets.QTableWidgetItem(data))
                self.objectPropertiesTable.scrollToBottom()
            rowCount += 1
        self.uniqueLabel.sort()

    def updatePosition(self, index):
        # update position of image in the list
        self.index = index
        img = self.images[self.index]
        self.positionLabel.setText(img.name)

    def startTracking(self):
        # creates an instance on another thread, modul "tracking", where we are tracking the movement of objects
        self.running_thread = Tracking(self.images, self.index, self.objectProperties, self.neighborhoodSize, self.filter, self.threshold, self.kernel
            , self.sigma, self.sigmaColor, self.sigmaSpatial, self.segmentationTech, self.manThreshold, self.sigmaCanny
            , int(self.neighborhoodWindow.text()), int(self.neighborhoodWindow2.text()), self.lowThresh, self.highThresh, self.size, self.sizeKernel, self.autoMethod)

        # checkigng if the values from UI are suitable
        try:
            valueX = int(self.neighborhoodWindow.text())
            if valueX < 0 :
                self.errorWarning.setText("The size of neighborhood has to have positive integer value!!")
                self.errorWarning.setWindowTitle("Invalid Value")
                return self.popup()
        except ValueError:
            self.errorWarning.setText("You have to enter positive integer value!")
            return self.popup()

        try:
            valueY = int(self.neighborhoodWindow2.text())
            if valueY < 0:
                self.errorWarning.setText("The size of neighborhood has to have positive integer value!!")
                self.errorWarning.setWindowTitle("Invalid Value")
                return self.popup()
        except ValueError:
            self.errorWarning.setText("You have to enter positive integer value!")
            return self.popup()

        # enabling/disabling stop/start buttons and table buttons
        self.startTrackingButton.setEnabled(False)
        self.stopTrackingButton.setEnabled(True)
        self.tableButtonsGroupBox.setEnabled(False)
        self.parametersWindow.setEnabled(False)
        self.neighborhoodWindow.setEnabled(False)
        self.neighborhoodWindow2.setEnabled(False)
        self.pathsList.setEnabled(False)
        self.backgroundGroupBox.setEnabled(False)
        self.filterGroupBox.setEnabled(False)
        self.segmentationGroupBox.setEnabled(False)
        self.contrastGroupBox.setEnabled(False)
        self.histogramMenuButton.setEnabled(False)
        self.openMenuButton.setEnabled(False)
        self.saveMenuButton.setEnabled(False)
        self.labelsButton.setEnabled(False)

        # starting the computing thread and connecting emited results from the thread with specific methods
        self.running_thread.resultImage.connect(self.showImg)
        self.running_thread.resultTable.connect(self.tableUpdate)
        self.running_thread.currentIndex.connect(self.updatePosition)
        self.running_thread.errorEmit.connect(self.raiseErrorMessage)
        self.running_thread.start()

    def raiseErrorMessage(self, error):
        # raising error if we don't find any object in the enlarged window
        if error:
            self.stopTracking()
            self.errorWarning.setText("There is no fiber in the enlarged window")
            self.errorWarning.setWindowTitle("Warning")
            self.popup(icon = 2)

    def stopTracking(self):
        # stops the computing when we press stop button in UI
        if Tracking.sameLabelAlive:
            self.running_thread.stop()

            self.startTrackingButton.setEnabled(True)
            self.stopTrackingButton.setEnabled(False)
            self.tableButtonsGroupBox.setEnabled(True)
            self.running_thread = None

            self.parametersWindow.setEnabled(True)
            self.neighborhoodWindow.setEnabled(True)
            self.neighborhoodWindow2.setEnabled(True)
            self.pathsList.setEnabled(True)
            self.backgroundGroupBox.setEnabled(True)
            self.filterGroupBox.setEnabled(True)
            self.segmentationGroupBox.setEnabled(True)
            self.contrastGroupBox.setEnabled(True)
            self.histogramMenuButton.setEnabled(True)
            self.openMenuButton.setEnabled(True)
            self.saveMenuButton.setEnabled(True)
            self.labelsButton.setEnabled(True)

    def sortTable(self):
        # sorts the items in the table
        self.objectPropertiesTable.sortItems(0)
        self.filterTableButton.setEnabled(True)
        self.plotTableButton.setEnabled(True)
        self.plotOrientationButton.setEnabled(True)
        self.deleteTableButton.setEnabled(True)

    def deleteRow(self, label = None):
        # deletes selected row/s from the table
        indices = None
        rowCount = None

        if label == False:
            # getting the indices of the selected rows
            indices = self.objectPropertiesTable.selectionModel().selectedRows()
            for ind in indices:
                fiber = int(self.objectPropertiesTable.item(ind.row(),0 ).text())
                rowCount = ind.row()
                for label in self.uniqueLabel:
                    if label < fiber:
                        rowCount -= self.labelCount[label]
                # delete the data from the positions list
                del self.pozX[fiber][rowCount]
                del self.pozY[fiber][rowCount]
                self.objectPropertiesTable.removeRow(ind.row())

        # delete rows with specific label number
        else:
            indices = self.objectPropertiesTable.findItems(str(label), QtCore.Qt.MatchExactly)
            for index in indices:
                self.objectPropertiesTable.removeRow(index.row())

    def exportTable(self):
        # eport the table into .xlsx format and saves it into specific diretorz
        filename, _ = QFileDialog.getSaveFileName(self, 'Save Table', "C:/Users/Michal/Documents/Škola/Diplomka/src", '*.xlsx')
        if not filename:
            return True
        wbk = Workbook()
        ws = wbk.active
        for col in range(self.objectPropertiesTable.columnCount()):
            for row in range(self.objectPropertiesTable.rowCount()):
                    teext = str(self.objectPropertiesTable.item(row, col).text())
                    ws.cell(row = row+1 , column=col+1).value=teext
        wbk.save(filename)

    def saveImage(self):
        # save the currently dislayed image into specific directory
        files_types = "JPG (*.jpg);;PNG (*.png);;TIF(*.tif)"
        filename, _ = QFileDialog.getSaveFileName(self, 'Save Image', "C:", files_types)
        if not filename:
            return True
        self.pixmap.save(filename)

    def filterTable(self):
        # perform filtration of table data based on centroid positions
        deleteList = []
        # check if any number of records for specific label in the table is not lower than 20 and gives posibility to delete it
        for label in self.uniqueLabel:
            if self.labelCount[label] < 20:
                self.errorWarning.setText(f"The number of records ({self.labelCount[label]}) for fiber with label {label} is less than 20.")
                self.errorWarning.setInformativeText("Do you want to delete records of this label?")
                self.errorWarning.setWindowTitle("Warning")
                self.errorWarning.setStandardButtons(QtWidgets.QMessageBox.Yes|QtWidgets.QMessageBox.No)
                self.errorWarning.setDefaultButton(QtWidgets.QMessageBox.Yes)

                result = self.popup(icon = 3)

                if result == QtWidgets.QMessageBox.Yes:
                    deleteList.append(label)
                    self.deleteRow(label = label)

        for x in deleteList:
            self.uniqueLabel.remove(x)

        predictionX = 0
        predictionY = 0
        x0 = 0
        x1 = 0
        currentRow = 0
        deleteList = []

        # we perform a interpolation of the given centroid position and comparing them with the positions in table
        # if the position in table is not in given tolerance we mark it "red" and user can delete it or keep it afterwards
        for label in self.uniqueLabel:
            for count,(x2,y2) in enumerate(zip(self.pozX[label],self.pozY[label])):
                if count > 1:
                    predictionX += x1 - x0
                    predictionNormX = np.divide(predictionX, count-1)

                    if abs(x1+predictionNormX-x2)<(predictionNormX*0.2) and self.objectPropertiesTable.item(count + currentRow,2) != None:
                        self.objectPropertiesTable.item(count + currentRow,2).setBackground(QColor('yellow'))
                    elif self.objectPropertiesTable.item(count + currentRow,2) != None:
                        if label not in deleteList: deleteList.append(label)
                        self.objectPropertiesTable.item(count + currentRow,2).setBackground(QColor('red'))
                    x0 = x1
                    x1 = x2

                    predictionY += y1 - y0
                    predictionNormY = np.divide(predictionY, count-1)

                    if abs(y1+predictionNormY-y2)<(predictionNormY*0.5) and self.objectPropertiesTable.item(count + currentRow,2) != None:
                        self.objectPropertiesTable.item(count + currentRow,3).setBackground(QColor('yellow'))
                    elif self.objectPropertiesTable.item(count + currentRow,2) != None:
                        if label not in deleteList: deleteList.append(label)
                        self.objectPropertiesTable.item(count + currentRow,3).setBackground(QColor('red'))
                    y0 = y1
                    y1 = y2
                elif count == 0:
                    x0 = x2
                    y0 = y2
                    self.objectPropertiesTable.item(count + currentRow,2).setBackground(QColor('yellow'))
                    self.objectPropertiesTable.item(count + currentRow,3).setBackground(QColor('yellow'))
                elif count == 1:
                    x1 = x2
                    y1 = y2
                    self.objectPropertiesTable.item(count + currentRow,2).setBackground(QColor('yellow'))
                    self.objectPropertiesTable.item(count + currentRow,3).setBackground(QColor('yellow'))

            currentRow += self.labelCount[label]
            predictionX = 0
            predictionY = 0

        deleteLabel = []
        # posibility to delete data of labels where the values were not in the tolerance
        for label in deleteList:
                self.errorWarning.setText(f"Mismatching values were found in records of fiber with label {label}.")
                self.errorWarning.setWindowTitle("Warning")
                self.errorWarning.setStandardButtons(QtWidgets.QMessageBox.Yes|QtWidgets.QMessageBox.No)
                self.errorWarning.setDefaultButton(QtWidgets.QMessageBox.Yes)

                result = self.popup(icon = 3)

                if result == QtWidgets.QMessageBox.Yes:
                    deleteLabel.append(label)
                    self.deleteRow(label = label)
        self.errorWarning.setInformativeText(None)

        # update the label list in the table
        for x in deleteLabel:
            self.uniqueLabel.remove(x)

    def plotResutlts(self):
        # perform cubic spline interpolation of the data
        problemLabel = []
        for label in self.uniqueLabel:
            if self.labelCount[label] <= 3:
                problemLabel.append(label)
                continue
            fig, ax = plt.subplots(1,1, figsize = (8,6))
            axes= plt.gca()
            # changing the axis range and position
            axes.set_xlim([0,1024])
            maxTick = None
            minTick = None

            if min(self.pozY[label]) -50 < 0:
                maxTick = 0
            else:
                maxTick = min(self.pozY[label]) -50
            if max(self.pozY[label]) + 50 > 1024:
                minTick = 1024
            else:
                minTick = max(self.pozY[label]) + 50

            axes.set_ylim([minTick,maxTick])
            axes.xaxis.tick_top()
            axes.xaxis.set_label_position('top')
            axes.yaxis.tick_left()
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title('Průběh trajektorie vlákna')
            points = np.array([self.pozX[label],self.pozY[label]]).T
            distance = np.cumsum( np.sqrt(np.sum( np.diff(points, axis=0)**2, axis=1 )) )
            distance = np.insert(distance, 0, 0)/distance[-1]
            alpha = np.linspace(0, 1, 200)

            splineX = interp1d(distance, points, kind = 'cubic', axis = 0)
            interpolated_points = splineX(alpha)

            #fitting a curve into the points
            z = np.polyfit(self.pozX[label],self.pozY[label],3)
            fit = np.poly1d(z)
            x_new = np.linspace(self.pozX[label][0], self.pozX[label][-1], 200)
            y_new = fit(x_new)

            plt.plot(*points.T,'bx', label="Body těžiště")
            plt.plot(*interpolated_points.T,'r-', label="Interpolační křivka")
            plt.plot(x_new, y_new, 'g--', label = "Aproximační křivka")
            plt.legend(loc='best')

        for label in problemLabel:
            self.errorWarning.setText(f"There is not enought records to plot label {label}'s trajectory.")
            self.errorWarning.setWindowTitle("Warning")

            result = self.popup(icon = 3)

        # show the plot
        plt.show()

    def plotOrientation(self):
        problemLabel = []
        for i, label in enumerate(self.uniqueLabel):
            if self.labelCount[label] <= 3:
                problemLabel.append(label)
                continue
            fig, ax = plt.subplots(1,1, figsize = (8,6))
            unit = 0.5
            yTick = np.arange (0, 2+unit, unit)
            yLabel = [r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$",  r"$2\pi}$"]
            axes = plt.gca()
            axes.set_ylim([0,2*math.pi])
            axes.set_yticks(yTick*np.pi)
            plt.xlabel('x')
            plt.ylabel('orientace')
            plt.title('Průběh orientace a elongace vlákna')
            ax.set_yticklabels(yLabel)

            splineX = interp1d(self.pozX[label], self.orientation[label], kind = 'linear')
            newX = np.linspace(min(self.pozX[label]),max(self.pozX[label]), 200)
            interpolated_points = splineX(newX)
            ax.plot(self.pozX[label], self.orientation[label], 'bo',label="Hodnoty orientace")
            ax.plot(newX, interpolated_points,'-r', label="Interpolační křivka orientace")

            ax2 = ax.twinx()
            ax2.set_ylabel('elongace')
            splineX2 = interp1d(self.pozX[label], self.ellongation[label], kind = 'linear')
            interpolated_points2 = splineX2(newX)
            ax2.set_ylim([0,5])
            yTick = np.arange(0,6,1)
            ax2.set_yticks(yTick)

            ax2.plot(self.pozX[label], self.ellongation[label], 'kx', label = "Hodnoty elongace")
            ax.plot(np.nan, 'kx', label = 'Hodnoty elongace')
            ax2.plot(newX, interpolated_points2, '--g', label = "Interpolační křivka elongace")
            ax.plot(np.nan, '--g', label = 'Interpolační křivka elongace')

            ax.legend(loc='best', fontsize = 'small')

        for label in problemLabel:
            self.errorWarning.setText(f"There is not enought records to plot label {label}'s orientation and elongation.")
            self.errorWarning.setWindowTitle("Warning")

            result = self.popup(icon = 3)
        fig.tight_layout()

        plt.show()

    def closeApp(self):
        # closing the app
        self.errorWarning.setText("Do you want to close the application?")
        self.errorWarning.setWindowTitle("Closing app")
        self.errorWarning.setStandardButtons(QtWidgets.QMessageBox.Yes|QtWidgets.QMessageBox.No)
        self.errorWarning.setDefaultButton(QtWidgets.QMessageBox.Yes)

        result = self.popup(icon = 3)

        if result == QtWidgets.QMessageBox.Yes:
            self.close()
