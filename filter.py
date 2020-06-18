from skimage.restoration import denoise_bilateral
import skimage
import numpy as np
from scipy import signal
import scipy

# methods used for filtering image using certain algorithm
class medianFilter():

    @staticmethod
    def process(img, kernel):
        imageFilt = skimage.filters.median(img, kernel)
        return imageFilt


class meanFilter():

    @staticmethod
    def process(img, kernel):
        kernel1 = np.ones((kernel, kernel), dtype=np.int8)

        imageFilt = np.divide(signal.convolve2d(
            img, kernel, boundary='symm'), kernel.size)
        return imageFilt


class gaussFilter():

    @staticmethod
    def process(img, sigma):
        imageFilt = skimage.filters.gaussian(
            img, sigma=sigma, preserve_range=True)
        return imageFilt


class bilateralFilter():

    @staticmethod
    def process(img, sigmaC, sigmaS):
        imageFilt = skimage.restoration.denoise_bilateral(
            img, sigma_color=sigmaC, sigma_spatial=sigmaS)
        return imageFilt

class maxFilter():

    @staticmethod
    def process(img, kernel):
        imageFilt = scipy.ndimage.minimum_filter(img, size=kernel)
        return imageFilt
