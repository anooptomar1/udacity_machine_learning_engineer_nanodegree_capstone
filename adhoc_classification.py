
from evaluator import Evaluator
from sketch_classifier import *
from sketch_recognition_trainer import SketchRecognitionClassifier
from misc_utils import MiscUtils


def get_sketch_recogniser():
    return SketchRecognitionClassifier(
        params_filename=TRAINED_PARAMS_FILENAME,
        cookbook_filename=TRAINED_COOKBOOK_FILENAME,
        classifier_filename=TRAINED_CLASSIFIER_FILENAME
    )

    # return SketchRecognitionClassifier(
    #     params_filename=WEB_PARAMS_FILENAME,
    #     cookbook_filename=WEB_COOKBOOK_FILENAME,
    #     classifier_filename=WEB_CLASSIFIER_FILENAME
    # )


if __name__ == '__main__':
    print __file__

    sr = get_sketch_recogniser()

    test_image = "../png/bicycle/1684.png"
    test_image_2 = "../png/face/6241.png"
    img = MiscUtils.crop_and_rescale_image(MiscUtils.load_image(test_image))
    img2 = MiscUtils.crop_and_rescale_image(MiscUtils.load_image(test_image_2))

    print sr.predict([img, img2])


