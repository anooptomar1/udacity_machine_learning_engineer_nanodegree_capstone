
from evaluator import Evaluator
from sketch_classifier import *
from sketch_recognition_trainer import SketchRecognitionClassifier
from tuning_helper import *


def get_sketch_recogniser():
    return SketchRecognitionClassifier(
        params_filename=TRAINED_PARAMS_FILENAME,
        cookbook_filename=TRAINED_COOKBOOK_FILENAME,
        classifier_filename=TRAINED_CLASSIFIER_FILENAME
    )


if __name__ == '__main__':
    print __file__

    train_labels, train_images, test_labels, test_images = get_training_and_test_data()