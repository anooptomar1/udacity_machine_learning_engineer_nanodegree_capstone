from tuning_helper import *
from evaluators.evaluator import Evaluator
from sketch_recognition_trainer import FeatureExtractorKeys
from sketch_recognition_trainer import MeanShiftSketchRecognitionTrainer
from sketch_recognition_trainer import ParamKeys
from sketch_recognition_trainer import SketchRecognitionTrainer
from sketch_recognition_trainer import SketchRecognitionClassifier
import numpy as np
import time
from classifiers.sketch_classifier import *


def evaluate_classifiers(classifiers):
    train_labels, train_images, test_labels, test_images = get_training_and_test_data()

    selected_labels = list(set(train_labels))

    params = build_params(num_classes=len(selected_labels),
                          training_size=len(train_images),
                          test_size=len(test_images),
                          feature_extractor=FeatureExtractorKeys.SIFT,
                          window_resolution=0.125,
                          window_overlap=2.5,
                          num_clusters=400,
                          image_size=256)

    sketch_classifier = SketchRecognitionClassifier()
    sketch_classifier.load_clustering_and_cookbook_from_file(TRAIN_COOKBOOK_FILENAME)

    train_images_codelabels = np.load(TRAIN_CODE_LABELS_FILENAME)

    # to ensure consistency
    train_labels = np.load(TRAIN_LABELS_FILENAME)

    print "creating codelabels for test images"
    test_images_codelabels = sketch_classifier.code_labels_for_image_descriptors(
        sketch_classifier.extract_image_descriptors(test_images)
    )

    for clf_wrapper in classifiers:
        evaluate_classifier(
            clf_wrapper,
            params,
            train_images_codelabels, train_labels,
            test_images_codelabels, test_labels
        )


def evaluate_classifier(clf_wrapper,
                        params,
                        train_images_codelabels, train_labels,
                        test_images_codelabels, test_labels):

    print "==\nevaluate_classifier (test size: {})\n{}".format(len(test_labels), clf_wrapper)

    print "training classifier {}".format(clf_wrapper.clf)
    start_time = time.time()
    clf_wrapper.fit(X=train_images_codelabels, labels=train_labels)
    et = (time.time() - start_time) * 1000.0
    print "finished training classifier - took {}ms".format(et)

    # evaluate
    print "proceeding to evaluate classifier on test set {}".format(len(test_labels))
    encoded_test_labels = clf_wrapper.le.transform(test_labels)

    evaluator = Evaluator(
        clf=clf_wrapper.clf,
        label_encoder=clf_wrapper.le,
        params=params,
        output_filepath="../results/evaluation_results_{}.json".format(clf_wrapper)
    )

    evaluator.results["classifier"] = "{}".format(clf_wrapper.clf)
    evaluator.results["classifier_training_time"] = "{}".format(et)

    evaluation_results = evaluator.evaluate(X=test_images_codelabels, y=encoded_test_labels)
    print evaluation_results


if __name__ == '__main__':
    print __file__

    classifiers = [
        # MultinomialNaiveBayesSketchClassifier(
        #     filename="../data/MultinomialNaiveBayesSketchClassifier_1.dat"
        # ),
        # GaussianNaiveBayesSketchClassifier(
        #     filename="../data/GaussianNaiveBayesSketchClassifier_1.dat"
        # ),
        # KNeighborsClassifierSketchClassifier(
        #     filename="../data/KNeighborsClassifierSketchClassifier_Grid_1.dat",
        #     n_neighbors=10
        # ),
        # SVCSketchClassifier(
        #     filename="../data/SVCSketchClassifier_1.dat"
        # ),
        LinearSVCSketchClassifier(
            filename="../data/LinearSVCSketchClassifier_1.dat"
        )
    ]

    #evaluate_classifier(LinearSVCSketchClassifier(filename="../data/LinearSVCSketchClassifier_1.dat"))

    evaluate_classifiers(classifiers)


