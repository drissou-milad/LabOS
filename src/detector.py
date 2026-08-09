from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max
import numpy as np

from src import config


class CellDetector:

    def __init__(
        self,
        sigma=config.GAUSSIAN_SIGMA,
        threshold_abs=config.DETECTION_THRESHOLD,
        min_distance=config.CELL_RADIUS,
    ):
        self.sigma = sigma
        self.threshold_abs = threshold_abs
        self.min_distance = min_distance

    def detect(self, image):

        image = gaussian_filter(image, sigma=self.sigma)

        centers = peak_local_max(
            image,
            threshold_abs=self.threshold_abs,
            min_distance=self.min_distance,
        )

        return centers

    def detect_volume(self, volume):

        detections = []

        for frame in range(volume.shape[0]):

            mip = np.array(volume[frame]).max(axis=0)

            centers = self.detect(mip)

            detections.append(centers)

        return detections