
from skimage.feature import hog
import math
import numpy as np
import cv2
from feature_extractor import BaseFeatureExtractor

class SKImageHOGF2eatureExtractor(BaseFeatureExtractor):

    def __init__(self, params):
        super(SKImageHOGF2eatureExtractor, self).__init__(params)

        if params is not None and "window_ratio" in params:
            self.window_ratio = params["window_ratio"]
        else:
            self.window_ratio = 0.25

        if params is not None and "cell_res" in params:
            self.cell_xres = params["cell_res"]
            self.cell_yres = params["cell_res"]
        else:
            self.cell_xres = 4
            self.cell_yres = 4

        if params is not None and "cell_orient" in params:
            self.cell_orient = params["cell_orient"]
        else:
            self.cell_orient = 8

        self.dense_detector = None

    def _compute(self, image):
        pixels_per_window = self.window_ratio * image.shape[0]

        #size = self.feature_count, self.cell_xres * self.cell_yres * self.cell_orient
        #descriptor = np.zeros(size, dtype=np.float)

        # descriptors = []
        descriptors = None

        y = 0
        while y < image.shape[0] - pixels_per_window/2:
            x = 0
            while x < image.shape[1] - pixels_per_window/2:
                x2 = x + pixels_per_window
                y2 = y + pixels_per_window

                # tmp = cv2.rectangle(image, (int(x), int(y)), (int(x2), int(y2)), 0, 1)
                # cv2.imshow('Image', image)
                # cv2.waitKey()

                # cv2.imshow('image', image[y:y2, x:x2])
                # cv2.waitKey()

                clipped_image = image[y:y2, x:x2]
                fd = hog(
                    clipped_image,
                    orientations=self.cell_orient,
                    pixels_per_cell=(pixels_per_window/self.cell_xres, pixels_per_window/self.cell_yres),
                    cells_per_block=(1, 1),
                    feature_vector=True,
                    visualise=False)

                # descriptors.append(fd)
                if descriptors is None:
                    descriptors = fd
                else:
                    descriptors = np.vstack((descriptors, fd))

                x += pixels_per_window/2

            y += pixels_per_window/2

        # return np.array(descriptors, dtype=np.float)
        return descriptors

    def _extract_keypoints(self, img):
        return self._get_dense_detector().detect(img)

    def _get_dense_detector(self):
        if self.dense_detector is not None:
            return self.dense_detector

        self.dense_detector = cv2.FeatureDetector_create("Dense")
        self.dense_detector.setInt("initXyStep", self.step_size)
        self.dense_detector.setInt("initFeatureScale", self.feature_scale)
        self.dense_detector.setInt("initImgBound", self.img_bound)

        return self.dense_detector


if __name__ == '__main__':
    input_image = cv2.imread('/Users/josh/Desktop/1.png')
    input_image_sift = np.copy(input_image)
    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    descriptors = SKImageHOGF2eatureExtractor(None)._compute(gray_image)
    print "descriptors shape {}".format(descriptors.shape)
