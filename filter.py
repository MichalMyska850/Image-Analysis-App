from skimage.restoration import denoise_bilateral
import skimage
import numpy as np
from scipy import signal

# methods used for filtering image using certain algorithm
class medianFilter():

    @staticmethod
    def process(img, kernel):
        imageFilt = skimage.filters.median(img, kernel)
        return imageFilt


class meanFilter():

    @staticmethod
    def process(img, kernel):
        #img = skimage.util.img_as_uint(img)
        kernel1 = np.ones((5, 5), dtype=np.int8)

        imageFilt = np.divide(signal.convolve2d(
            img, kernel, boundary='symm'), kernel.size)
        #imageFilt = skimage.filters.rank.mean(img, kernel)
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
