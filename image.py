import skimage
import numpy as np
from scipy import ndimage as ndi
from skimage import feature
import math
import matplotlib.pyplot as plt
from skimage.util import img_as_uint, img_as_bool

from collections.abc import Iterable
import numpy as np
from scipy import ndimage as ndi
from skimage import data
import matplotlib.pyplot as plt
import skimage

from skimage.util import img_as_float


class Image():

    def __init__(self, imagePath):
        super().__init__()
        self.imagePath = imagePath

        string = self.imagePath.split('/')
        self.name = string[-1]

        self.data = []
        self.medianBG = []
        self.meanBG = []
        self.divisionImg = []
        self.neighborhoodImages = []
        self.subtractImg = []
        self.thresholdedImage = []
        self.mask = []
        self.segmentImg = []
        self.sobelImage = []
        self.markers = []
        self.labelImage = []
        self.properties = []

    def loadImage(self):
        # import the image using its path
        self.data = skimage.util.img_as_float64(
            skimage.io.imread(self.imagePath))
        return self.data

    def contrast(self, black, white):
        # convert image to uint16
        image = skimage.util.img_as_uint(self.data)
        # rescale the intensity of image
        imageWithExposure = skimage.exposure.rescale_intensity(
            image, in_range=(black, white))
        return imageWithExposure

    def medianBackground(self, neighborhood):
        # create a median background
        for image in neighborhood:
            if image.data == []:
                image.loadImage()

        # find number of rows and columns of image
        row, col = self.data.shape

        self.neighborhoodImages = []
        # fill the inital list and convert it to 1D array rather than 2D
        for image in neighborhood:
            self.neighborhoodImages.append(image.data.flatten())

        # count median of images values
        self.medianBG = np.median(self.neighborhoodImages, axis=0)
        # reshape the array to 2D
        self.medianBG = np.asarray(np.reshape(self.medianBG, (row, col)))

        # subtract the median background from original image in abslute value
        subtract = np.abs(np.subtract(self.data, self.medianBG))

        # apply median filter
        kernel = skimage.morphology.square(5)
        self.subtractImg = skimage.filters.median(subtract, kernel)

        return self.subtractImg

    def threshold(self, thresh):

        # create thresholded image
        foregroundImage = self.subtractImg >= thresh
        self.thresholdedImage = foregroundImage * 1

        return self.thresholdedImage

    def dilation(self):

        kernel = skimage.morphology.square(5)
        # apply opening to get rid of small objects
        opening = skimage.morphology.opening(self.thresholdedImage, kernel)

        # fill any holes in objects
        filled = ndi.binary_fill_holes(opening) * 1

        # create a mask wih dilation
        self.mask = skimage.morphology.dilation(
            filled, skimage.morphology.square(3)) * 1

        return self.mask

    def meanBackground(self, neighborhood):

        row, col = self.data.shape

        # load images if they are not already
        for image in neighborhood:
            if image.data == []:
                image.loadImage()

        self.neighborhoodImages = []

        # fill the list of images in neighborhood and create 1D array rather than 2D
        for image in neighborhood:
            self.neighborhoodImages.append(image.data.flatten())

        # count mean of images values to create mean background
        self.meanBG = np.mean(self.neighborhoodImages, axis=0)

        # reshape the array to 2D
        self.meanBG = np.asarray(np.reshape(self.meanBG, (row, col)))

        # change the values of mean background to median background on positions where created mask == 0
        self.meanBG = np.where(self.mask == 0, self.meanBG, self.medianBG)

        return self.meanBG

    def imageDivision(self):

        row, col = self.data.shape

        # create 1D arrays
        image = self.data.flatten()
        imageBG = self.meanBG.flatten()

        # get rid of possible 0 values in the image
        imageBG = np.where(imageBG == 0, 0.001, imageBG)

        # divide the original image by median background
        self.divisionImg = np.asarray(np.reshape(
            np.divide(image, imageBG), (row, col)))

        return self.divisionImg

    def thresholdSegment(self, image, thresh):

        # create thresholded image with manualy acquired threshold
        foregroundImage = image <= thresh
        self.segmentImg = foregroundImage * 1

        return self.segmentImg

    def autoThresh(self, method, image):
        # apply one of the automatic thresholding method
        if method == 'Otsu':
            threshold = skimage.filters.threshold_otsu(image)
            foreground = image < threshold
            self.segmentImg = foreground * 1

        elif method == 'Yen':
            threshold = skimage.filters.threshold_yen(image)
            foreground = image < threshold
            self.segmentImg = foreground * 1

        elif method == 'Iso':
            threshold = skimage.filters.threshold_isodata(image)
            foreground = image < threshold
            self.segmentImg = foreground * 1

        elif method == 'Triangle':
            threshold = skimage.filters.threshold_triangle(image)
            foreground = image < threshold
            self.segmentImg = foreground * 1

        return self.segmentImg

    def edges(self, method, image):
        # create edge map with one of the specific methods
        if method == 'Sobel':
            self.sobelImage = skimage.filters.sobel(image, mask=self.mask)
        elif method == 'Laplace':
            self.sobelImage = skimage.filters.laplace(image, mask=self.mask)
        elif method == 'Prewitt':
            self.sobelImage = skimage.filters.prewitt(image, mask=self.mask)
        elif method == 'Roberts':
            self.sobelImage = skimage.filters.roberts(image, mask=self.mask)

        return self.sobelImage

    def cannyEdgeDetector(self, image, sigma, low, high):
        # create edge map using Canny Edge Detector on area limited by created mask
        self.segmentImg = skimage.feature.canny(
            image, sigma, low_threshold=low, high_threshold=high) * 1
        self.segmentImg = np.where(self.mask == 0, 0, self.segmentImg)
        return self.segmentImg

    def fillingHoles(self):
        # close edges created by Canny Edge Detector
        kernel = skimage.morphology.disk(7)
        self.segmentImg = skimage.morphology.closing(self.segmentImg, kernel)

        return self.segmentImg

    def edgesFilling(self):
        # filles the edges created by one of the methods
        filled = ndi.binary_fill_holes(self.sobelImage) * 1

        return filled

    def kmeans(self, img, segments, sigma, iterations, compact):

        self.segmentImg = skimage.segmentation.slic(
            img, n_segments=segments, sigma=sigma, compactness=compact, max_iter=iterations)
        return self.segmentImg

    def labelling(self):
        color = ('blue', 'cyan', 'darkorange', 'indigo',
                 'magenta', 'pink', 'red', 'yellow', 'yellowgreen')
        # fill holes in the objects
        filledHoles = ndi.binary_fill_holes(self.segmentImg)
        # remove objects smaller than 50 px
        labels = skimage.morphology.remove_small_objects(
            filledHoles, min_size=50)
        # clear borders of any unwanted pixels
        borders = skimage.segmentation.clear_border(labels)

        # label objects
        self.labelImage = skimage.measure.label(borders, background=0)

        # create colored image based on the performed labeling
        colorLabeledImage = skimage.color.label2rgb(
            self.labelImage, colors=color, bg_label=0)

        # create a list of properties of diffrent objects
        self.properties = skimage.measure.regionprops(self.labelImage)

        return self.properties, self.labelImage, colorLabeledImage
