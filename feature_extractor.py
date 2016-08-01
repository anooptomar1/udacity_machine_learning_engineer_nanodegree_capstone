"""
Responsible for extracting descriptors for a given image 
"""

import time
import cv2
from constants import *


class BaseFeatureExtractor(object):

    def __init__(self, params):
        self.compute_count = 0.
        self.compute_total_time = 0.
        self._image_size = 256.

    @property
    def image_size(self):
        return self._image_size

    def get_average_compute_time(self):
        if self.compute_count == 0:
            return 0

        return self.compute_total_time / self.compute_count

    def compute(self, image):
        self._image_size = image.shape[0]

        start_time = time.time()

        descriptor = self._compute(image)

        end_time = time.time()
        et = end_time - start_time
        self.compute_count += 1
        self.compute_total_time += et * 1000.0

        return descriptor

    def _compute(self, image):
        return None


class DenseFeatureExtractor(BaseFeatureExtractor):

    def __init__(self, params):
        super(DenseFeatureExtractor, self).__init__(params)
        self.keypoint_shape = None
        self.keypoints = None
        self.img_bound = 5

        if params is not None and ParamKeys.WINDOW_RATIO in params:
            self.window_ratio = params[ParamKeys.WINDOW_RATIO]
        else:
            self.window_ratio = 0.125

        if params is not None and ParamKeys.WINDOW_OVERLAP in params:
            self.window_overlap = float(params[ParamKeys.WINDOW_OVERLAP])
        else:
            self.window_overlap = 2.5

        self.step_size = int(self.image_size * self.window_ratio / self.window_overlap)
        self.feature_scale = int(self.image_size * self.window_ratio)

        self.dense_detector = None

    def _extract_keypoints(self, img):
        if self.keypoint_shape is not None:
            if self.keypoint_shape[0] == img.shape[0] and self.keypoint_shape[1] == \
                    img.shape[1]:
                return self.keypoints

        self.keypoints = self._get_dense_detector().detect(img)

        self.keypoint_shape = img.shape

        return self.keypoints

    def _get_dense_detector(self):
        if self.dense_detector is not None:
            return self.dense_detector

        self.dense_detector = cv2.FeatureDetector_create("Dense")
        self.dense_detector.setInt("initXyStep", self.step_size)
        self.dense_detector.setInt("initFeatureScale", self.feature_scale)
        self.dense_detector.setInt("initImgBound", self.img_bound)

        return self.dense_detector


class SiftFeatureExtractor(DenseFeatureExtractor):

    def __init__(self, params):
        super(SiftFeatureExtractor, self).__init__(params)

    def _compute(self, image):
        return self._extract_image_features(img=image)

    def _extract_image_features(self, img):
        kps, des = cv2.SIFT().compute(img, self._extract_keypoints(img))
        return des


if __name__ == '__main__':
    print(__file__)

    import numpy as np

    #test_image = "../processed_png/airplane/airplane_0.png"
    test_image = "../processed_png/face/face_0.png"

    input_image = cv2.imread(test_image)
    input_image_sift = np.copy(input_image)
    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    fe = SiftFeatureExtractor(None)
    keypoints = fe._extract_keypoints(gray_image)
    print "keypoints len {}".format(len(keypoints))
    #input_image = cv2.drawKeypoints(input_image, keypoints, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    one_added = False
    for kp in keypoints:
        cx = int(kp.pt[0])
        cy = int(kp.pt[1])

        x = cx - (fe.feature_scale/2)
        y = cy - (fe.feature_scale / 2)
        x2 = x + fe.feature_scale
        y2 = y + fe.feature_scale

        #tmp = cv2.rectangle(input_image, (int(x), int(y)), (int(x2), int(y2)), 0, 1)
        tmp = cv2.circle(input_image, (cx, cy), 2, (0, 0, 255), -1)


    cv2.imshow('Dense feature detector', input_image)
    cv2.waitKey()
