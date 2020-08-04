from PyQt5.QtCore import pyqtSlot, pyqtSignal, QThread
import time
from filter import medianFilter, meanFilter, gaussFilter, bilateralFilter
import numpy as np
import skimage
import math


class Tracking(QThread):

    resultTable = pyqtSignal(object, object, object)
    resultImage = pyqtSignal(object)
    currentIndex = pyqtSignal(object)
    errorEmit = pyqtSignal(object)
    zeroCross = []
    sameLabelAlive = None
    propertiesNew = None

    def __init__(self, images, index, properties, neigbrhoodSize, filterType, thresh, kernel, sigma, sigmaC, sigmaS,
                 segmentType, manThreshold, sigmaCanny, windowSizeX, windowSizeY, lowThresh, highThresh, size, kernelSize, method):

        QThread.__init__(self)
        self.images = images
        self.index = index

        self.filter = filterType
        self.neigbrhoodSize = neigbrhoodSize
        self.threshold = thresh
        self.kernel = kernel
        self.sigma = sigma
        self.sigmaColor = sigmaC
        self.sigmaSpatial = sigmaS
        self.segmentationTech = segmentType
        self.manThreshold = manThreshold
        self.sigmaCanny = sigmaCanny
        self.widowSizeX = windowSizeX
        self.windowSizeY = windowSizeY
        self.lowThresh = lowThresh
        self.highThresh = highThresh
        self.dividedFiber = True
        self.size = size
        self.kernelSize = kernelSize
        self.autoMethod = method

        self.color = ('blue', 'cyan', 'darkorange', 'indigo',
                      'magenta', 'pink', 'red', 'yellow', 'yellowgreen')
        self.alive = True
        self.properties = properties
        self.changedThresh = manThreshold

    def run(self):
        if Tracking.sameLabelAlive == None:
            currentImage, currentImageProperties, orientation, ellongation = self.sameLabel()

        # checking if the button stop wasn't pressed
        if self.alive and Tracking.sameLabelAlive == None:
            self.index += 1
            error = False
            # emiting results from second thread to the UI
            self.resultImage.emit(currentImage)
            self.resultTable.emit(currentImageProperties,
                                  orientation, ellongation)
            self.currentIndex.emit(self.index)
            self.errorEmit.emit(error)
            Tracking.sameLabelAlive = True

        elif not self.alive and Tracking.sameLabelAlive == None:
            error = True
            self.errorEmit.emit(error)
            Tracking.sameLabelAlive = True

        while self.alive:
            self.dividedFiber = True
            # checks if the objects keeps together
            while self.dividedFiber:
                values = self.positionPredic()
            self.index += 1
            self.resultImage.emit(values[0])
            self.resultTable.emit(values[1], values[2], values[3])
            self.currentIndex.emit(self.index)

    def sameLabel(self):
        # preparing new image and init parameters
        labeledImageProperties, labeledImage, _ = self.binary()
        const = 10
        labels = []
        newLabels = []

        for objectLabel in self.properties:
            labels.append(objectLabel.label)

        for newObjectLabel in labeledImageProperties:
            newLabels.append(newObjectLabel.label)
        maxRow, maxCol = labeledImage.shape

        # iteration over all objects
        for fiber in self.properties:
            window = fiber.bbox
            if window[0] - self.windowSizeY < 0:
                minRowBound = 0
            else:
                minRowBound = window[0] - self.windowSizeY

            if window[2] + self.windowSizeY > maxRow:
                maxRowBound = maxRow
            else:
                maxRowBound = window[2] + self.windowSizeY

            if window[1] - self.widowSizeX < 0:
                minColBound = 0
            else:
                minColBound = window[1] - self.widowSizeX

            if window[3] + self.widowSizeX > maxCol:
                maxColBound = maxCol
            else:
                maxColBound = window[3] + self.widowSizeX
            # finding unique values in bounding box of every object in the new image
            unique = np.unique(
                labeledImage[minRowBound: maxRowBound, minColBound: maxColBound])
            # remove 0 from the unique - there will always be 0 due to black background
            if len(unique) != 0:
                unique = np.delete(unique, 0)
            if len(unique) != 1:
                self.alive = False

            # iteration over all unique numbers and changing them to their previous values
            # meaning, we want the object that are in the bounding box to have the same label as the object we
            # took the bounding box of
            for i, label in enumerate(unique):

                if fiber.label != unique[i] and fiber.label in labeledImage and i == 0:
                    # if the label != with unique we want to give it new unique label so we give it for now len(labels) + const
                    labeledImage = np.where(labeledImage == fiber.label,
                                            len(labels) + const, labeledImage)
                    const += 1
                    # we create new unique without 0
                    unique = np.delete(np.unique(
                        labeledImage[window[0] - self.windowSizeY:window[2] + self.windowSizeY, window[1] - self.widowSizeX:window[3] + self.widowSizeX]), 0)
                # for other uniques we change them to the same label
                labeledImage = np.where(
                    labeledImage == unique[i], fiber.label, labeledImage)

        n = np.abs(np.subtract(len(labels), len(newLabels)))

        labeledImage = skimage.morphology.remove_small_objects(
            labeledImage, min_size=self.size)
        # changing the labels that were change to len(label) + const to the end of current labels
        for i in range(10, const):
            labeledImage = np.where(labeledImage == len(
                labels) + i, max(labels) + n, labeledImage)
            n += 1
        # creating color image
        colorLabel = skimage.color.label2rgb(
            labeledImage, colors=self.color, bg_label=0)
        # count properties of objects
        Tracking.propertiesNew = skimage.measure.regionprops(labeledImage)

        labels = []
        newLabels = []

        for objectLabel in self.properties:
            labels.append(objectLabel.label)

        for newObjectLabel in Tracking.propertiesNew:
            newLabels.append(newObjectLabel.label)

        intersection = np.intersect1d(labels, newLabels)

        indices = [labels.index(x) for x in intersection]
        indicesNew = [newLabels.index(x) for x in intersection]
        orientation = [0] * len(Tracking.propertiesNew)

        # count orientation of object because originaly we get values from -PI/2 to PI/2 but we want the values from 0 to 2PI
        # parameter Tracking.zereCross include the labels which cross 0
        for index, indexNew in zip(indices, indicesNew):
            if Tracking.propertiesNew[indexNew].orientation + math.pi / 2 > (math.pi / 2) and self.properties[index].orientation + math.pi / 2 < (math.pi / 2) and (Tracking.propertiesNew[indexNew].orientation + math.pi / 2) - (self.properties[index].orientation + math.pi / 2) > math.pi / 2:

                if newLabels[indexNew] in Tracking.zeroCross:
                    Tracking.zeroCross.remove(newLabels[indexNew])
                else:
                    orientation[indexNew] = np.round(
                        Tracking.propertiesNew[indexNew].orientation + 3 * math.pi / 2, decimals=2)
                    Tracking.zeroCross.append(newLabels[indexNew])

            elif Tracking.propertiesNew[indexNew].orientation + math.pi / 2 < (math.pi / 2) and self.properties[index].orientation + math.pi / 2 > (math.pi / 2) and (self.properties[index].orientation + math.pi / 2) - (Tracking.propertiesNew[indexNew].orientation + math.pi / 2) > math.pi / 2:

                if newLabels[indexNew] in Tracking.zeroCross:
                    Tracking.zeroCross.remove(newLabels[indexNew])
                else:
                    orientation[indexNew] = np.round(
                        Tracking.propertiesNew[indexNew].orientation + 3 * math.pi / 2, decimals=2)
                    Tracking.zeroCross.append(newLabels[indexNew])

            elif newLabels[indexNew] in Tracking.zeroCross:

                if Tracking.propertiesNew[indexNew].orientation + math.pi / 2 < math.pi / 2:
                    orientation[indexNew] = np.round(
                        Tracking.propertiesNew[indexNew].orientation + 3 * math.pi / 2, decimals=2)
                else:
                    orientation[indexNew] = np.round(
                        Tracking.propertiesNew[indexNew].orientation + 3 * math.pi / 2, decimals=2)

        # if the orientation of curtain label is not filled (is 0) we give it value we got from properties, unchanged
        for i, fiber in enumerate(Tracking.propertiesNew):
            if orientation[i] == 0:
                orientation[i] = np.round(
                    fiber.orientation + math.pi / 2, decimals=2)

        ellongation = [0] * len(Tracking.propertiesNew)
        for i, fiber in enumerate(Tracking.propertiesNew):
            majorA = fiber.major_axis_length
            minorA = fiber.minor_axis_length

            ellongation[i] = np.round(math.log2(majorA / minorA), decimals=2)

        return colorLabel, Tracking.propertiesNew, orientation, ellongation

    def positionPredic(self):
        # this process is similar to the one in method sameLabel
        labeledImageProperties, labeledImage, _ = self.binary()
        labels = []
        const = 10
        newLabels = []

        for objectLabel in self.properties:
            labels.append(objectLabel.label)

        for newObjectLabel in Tracking.propertiesNew:
            newLabels.append(newObjectLabel.label)

        intersection = np.intersect1d(labels, newLabels)

        indices = [labels.index(x) for x in intersection]
        indicesNew = [newLabels.index(x) for x in intersection]

        for index, indexNew in zip(indices, indicesNew):
            window = Tracking.propertiesNew[indexNew].bbox

            # only difference is here, were we get a centroid positions
            centroidY, centroidX = self.properties[index].centroid
            centroidYNew, centroidXNew = Tracking.propertiesNew[indexNew].centroid

            positionX = int(
                np.round(centroidXNew - centroidX, decimals=0))
            positionY = int(
                np.round(centroidYNew - centroidY, decimals=0))

            # and move the bounding bo windows by difference of those centroids
            unique = np.unique(
                labeledImage[window[0] + positionY:window[2] + positionY, window[1] + positionX:window[3] + positionX])
            if len(unique) != 0:
                unique = np.delete(unique, 0)
            # in case we have more objects in the window, we increase the threshold in case of manual threshold segmentation
            # that's because it might happend that the objects break into more parts
            if len(unique) > 1 and self.manThreshold <= 1.0 and self.segmentationTech == "Manual Threshold":
                self.manThreshold += 0.0005
                return True

            for i, label in enumerate(unique):

                if newLabels[indexNew] != unique[i] and newLabels[indexNew] in labeledImage and i == 0:
                    labeledImage = np.where(labeledImage == newLabels[indexNew],
                                            len(newLabels) + const, labeledImage)
                    const += 1
                    unique = np.delete(np.unique(
                        labeledImage[window[0] + positionY:window[2] + positionY, window[1] + positionX:window[3] + positionX]), 0)

                labeledImage = np.where(
                    labeledImage == unique[i], newLabels[indexNew], labeledImage)

        self.dividedFiber = False
        self.manThreshold = self.changedThresh
        labeledImage = skimage.morphology.remove_small_objects(
            labeledImage, min_size=self.size)

        n = np.abs(np.subtract(len(labeledImageProperties), len(newLabels)))

        for i in range(10, const):
            if len(newLabels) + i in labeledImage:
                labeledImage = np.where(labeledImage == len(
                    newLabels) + i, max(newLabels) + n, labeledImage)
                n += 1

        colorLabel = skimage.color.label2rgb(
            labeledImage, colors=self.color, bg_label=0)
        self.properties = Tracking.propertiesNew
        Tracking.propertiesNew = skimage.measure.regionprops(labeledImage)

        labels = []
        newLabels = []

        for objectLabel in self.properties:
            labels.append(objectLabel.label)

        for newObjectLabel in Tracking.propertiesNew:
            newLabels.append(newObjectLabel.label)

        intersection = np.intersect1d(labels, newLabels)

        indices = [labels.index(x) for x in intersection]
        indicesNew = [newLabels.index(x) for x in intersection]
        orientation = [0] * len(Tracking.propertiesNew)

        for index, indexNew in zip(indices, indicesNew):
            if Tracking.propertiesNew[indexNew].orientation + math.pi / 2 > math.pi / 2 and self.properties[index].orientation + math.pi / 2 < math.pi / 2 and (Tracking.propertiesNew[indexNew].orientation + math.pi / 2) - (self.properties[index].orientation + math.pi / 2) > math.pi / 2:
                if newLabels[indexNew] in Tracking.zeroCross:
                    Tracking.zeroCross.remove(newLabels[indexNew])
                else:
                    orientation[indexNew] = np.round(
                        Tracking.propertiesNew[indexNew].orientation + 3 * math.pi / 2, decimals=2)
                    Tracking.zeroCross.append(newLabels[indexNew])

            elif Tracking.propertiesNew[indexNew].orientation + math.pi / 2 < math.pi / 2 and self.properties[index].orientation + math.pi / 2 > math.pi / 2 and (self.properties[index].orientation + math.pi / 2) - (Tracking.propertiesNew[indexNew].orientation + math.pi / 2) > math.pi / 2:
                if newLabels[indexNew] in Tracking.zeroCross:
                    Tracking.zeroCross.remove(newLabels[indexNew])
                else:
                    orientation[indexNew] = np.round(
                        Tracking.propertiesNew[indexNew].orientation + 3 * math.pi / 2, decimals=2)
                    Tracking.zeroCross.append(newLabels[indexNew])

            elif newLabels[indexNew] in Tracking.zeroCross:
                if Tracking.propertiesNew[indexNew].orientation + math.pi / 2 < math.pi / 2:
                    orientation[indexNew] = np.round(
                        Tracking.propertiesNew[indexNew].orientation + 3 * math.pi / 2, decimals=2)
                else:
                    orientation[indexNew] = np.round(
                        Tracking.propertiesNew[indexNew].orientation + 3 * math.pi / 2, decimals=2)

        for i, fiber in enumerate(Tracking.propertiesNew):
            if orientation[i] == 0:
                orientation[i] = np.round(
                    fiber.orientation + math.pi / 2, decimals=2)

        ellongation = [0] * len(Tracking.propertiesNew)
        for i, fiber in enumerate(Tracking.propertiesNew):
            majorA = fiber.major_axis_length
            minorA = fiber.minor_axis_length

            ellongation[i] = np.round(math.log2(majorA / minorA), decimals=2)

        return colorLabel, Tracking.propertiesNew, orientation, ellongation

    # creates binary image with settings from preprocessing

    def binary(self):
        img = self.images[self.index + 1]

        img.loadImage()

        neighborhood = self.images[self.index - (self.neigbrhoodSize + 3):self.index - 3
                                   ] + self.images[self.index + 4:self.index + (self.neigbrhoodSize + 4)]

        # Background

        img.medianBackground(neighborhood)
        img.threshold(self.threshold)
        img.dilation()
        img.meanBackground(neighborhood)
        img.imageDivision()

        # filtering

        if self.filter == "Median Filter":
            imgFilt = medianFilter.process(img.divisionImg, self.kernel)

        elif self.filter == "Mean Filter":
            imgFilt = meanFilter.process(img.divisionImg, self.kernelSize)

        elif self.filter == "Gauss Filter":
            imgFilt = gaussFilter.process(img.divisionImg, self.sigma)

        elif self.filter == "Bilateral Filter":
            imgFilt = bilateralFilter.process(
                img.divisionImg, self.sigmaColor, self.sigmaSpatial)
        elif self.filter == "Maximum Filter":
            imgFilt = maxFilter.process(img.divisionImg, self.kernelSize)

        # segmentation

        if self.segmentationTech == "Manual Threshold":
            img.thresholdSegment(imgFilt, self.manThreshold)

        elif self.segmentationTech == "Automatic Thresh":
            img.autoThresh(self.autoMethod, imgFilt)

        elif self.segmentationTech == "Edge Operators":
            img.edges(self.autoMethod, imgFilt)

        elif self.segmentationTech == "Canny Edge Detector":

            img.cannyEdgeDetector(imgFilt, self.sigmaCanny,
                                  self.lowThresh, self.highThresh)
            img.fillingHoles()

        # Labelling

        return img.labelling(self.size)

    # triggered when button is pressed
    def stop(self):
        self.alive = False
        self.wait()
