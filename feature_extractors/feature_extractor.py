import time
import cv2


class BaseFeatureExtractor(object):

    def __init__(self, params):
        self.compute_count = 0.
        self.compute_total_time = 0.

    @property
    def image_size(self):
        return 256.

    def get_average_compute_time(self):
        if self.compute_count == 0:
            return 0

        return self.compute_total_time / self.compute_count

    def compute(self, image):
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

        if params is not None and "window_ratio" in params:
            self.window_ratio = params["window_ratio"]
        else:
            self.window_ratio = 0.125

        if params is not None and "window_overlap" in params:
            self.window_overlap = float(params["window_overlap"])
        else:
            self.window_overlap = 2.

        self.step_size = int(self.image_size * self.window_ratio / self.window_overlap)
        self.feature_scale = int(self.image_size * self.window_ratio)
        self.img_bound = 5

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



