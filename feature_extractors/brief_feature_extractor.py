
import numpy as np
import cv2
from feature_extractor import DenseFeatureExtractor


class BriefFeatureExtractor(DenseFeatureExtractor):
    """
    http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_brief/py_brief.html
    http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_brief/py_brief.html
    http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_orb/py_orb.html
    """
    def __init__(self, params):
        super(BriefFeatureExtractor, self).__init__(params)

    def _compute(self, image):
        return self._extract_image_features(img=image)

    def _extract_image_features(self, img):
        kps, des = cv2.DescriptorExtractor_create("BRIEF").compute(img, self._extract_keypoints(img))
        return des


if __name__ == '__main__':
    input_image = cv2.imread('/Users/josh/Desktop/1.png')
    input_image_sift = np.copy(input_image)
    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    # keypoints = SiftFeatureExtractor(None).extract_keypoints(gray_image)
    # print "keypoints len {}".format(len(keypoints))
    # input_image = cv2.drawKeypoints(input_image, keypoints, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # cv2.imshow('Dense feature detector', input_image)
    # cv2.waitKey()

    print BriefFeatureExtractor(None).extract_image_features(gray_image).shape

