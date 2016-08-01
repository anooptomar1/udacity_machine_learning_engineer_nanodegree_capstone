"""
Run this to rebuild and evaluate the classifier
NB: Perform performing each step, the script will check if the file exists and load instead
(and saves the associated file at the end of each step).
"""

from evaluator import Evaluator
from sketch_recognition_trainer import SketchRecognitionTrainer
from tuning.tuning_helper import *
from misc_utils import MiscUtils
from constants import *


def rebuild(train_labels, train_images, test_labels, test_images, params_prefix=None):
    selected_labels = list(set(train_labels))

    params = build_params(
        num_classes=len(selected_labels),
        training_size=len(train_images),
        test_size=len(test_images),
        window_resolution=0.125,
        window_overlap=2.5,
        num_clusters=400,
        image_size=256
    )

    if params_prefix is not None:
        params["prefix"] = params_prefix

    trainer = SketchRecognitionTrainer(
        file_path=SketchRecognitionTrainer.get_cookbook_filename_for_params(params=params),
        run_parallel_processors=True,
        params=params
    )

    classifier = trainer.train_and_build_classifier(train_labels, train_images)
    encoded_test_labels = classifier.label_encoder.transform(test_labels)

    test_images_codelabels = trainer.code_labels_for_image_descriptors(
        trainer.extract_image_descriptors(test_images)
    )

    evaluator = Evaluator(
        clf=classifier.clf,
        label_encoder=classifier.label_encoder,
        params=params,
        output_filepath=SketchRecognitionTrainer.get_evaluation_filename_for_params(params=params)
    )

    # add timings to output
    evaluator.results["timings"] = {}
    for key, value in trainer.timings.iteritems():
        evaluator.results["timings"][key] = value

    evaluation_results = evaluator.evaluate(X=test_images_codelabels, y=encoded_test_labels)

    print evaluation_results


def rebuild_with_subset():
    print("=== rebuild_with_subset ===")

    train_labels, train_images, test_labels, test_images = get_training_and_test_data()

    rebuild(train_labels, train_images, test_labels, test_images)

def rebuild_with_subset_b():
    print("=== rebuild_with_subset_b ===")

    train_labels, train_images, test_labels, test_images = get_training_and_test_data_b()

    rebuild(train_labels, train_images, test_labels, test_images, "leaveout")

if __name__ == '__main__':
    print __file__

    rebuild_with_subset_b()

