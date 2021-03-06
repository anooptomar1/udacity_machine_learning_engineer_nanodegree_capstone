import time

from evaluator import Evaluator
from sketch_classifier import *
from sketch_recognition_trainer import SketchRecognitionClassifier
from tuning_helper import *


def evaluate_classifiers(classifiers):
    train_labels, train_images, test_labels, test_images = get_training_and_test_data()

    selected_labels = list(set(train_labels))

    params = build_params(num_classes=len(selected_labels),
                          training_size=len(train_images),
                          test_size=len(test_images),
                          window_resolution=0.125,
                          window_overlap=2.5,
                          num_clusters=400,
                          image_size=256)

    # params_filename, cookbook_filename, classifier_filename
    sketch_classifier = SketchRecognitionClassifier(
        params_filename=TRAINED_PARAMS_FILENAME,
        cookbook_filename=TRAINED_COOKBOOK_FILENAME,
        classifier_filename=None
    )

    if not os.path.exists(TRAINED_CODELABELS_FILENAME):
        raise Exception('No codelabels created for training dataset')

    train_images_codelabels = np.load(TRAINED_CODELABELS_FILENAME)

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
    encoded_test_labels = clf_wrapper.label_encoder.transform(test_labels)

    evaluator = Evaluator(
        clf=clf_wrapper.clf,
        label_encoder=clf_wrapper.label_encoder,
        params=params,
        output_filepath="../results/evaluation_results_{}.json".format(clf_wrapper)
    )

    evaluator.results["classifier"] = "{}".format(clf_wrapper.clf)
    evaluator.results["classifier_training_time"] = "{}".format(et)

    evaluation_results = evaluator.evaluate(X=test_images_codelabels, y=encoded_test_labels)
    print evaluation_results


if __name__ == '__main__':
    print __file__

    version = "2"

    classifiers = [
        # MultinomialNaiveBayesSketchClassifier(
        #     filename="../data/MultinomialNaiveBayesSketchClassifier_{}.dat".format(version)
        # ),
        # GaussianNaiveBayesSketchClassifier(
        #     filename="../data/GaussianNaiveBayesSketchClassifier_{}.dat".format(version)
        # ),
        KNeighborsClassifierSketchClassifier(
            filename="../data/KNeighborsClassifierSketchClassifier_Grid_{}.dat".format(version),
            n_neighbors=10
        ),
        # SVCSketchClassifier(
        #     filename="../data/SVCSketchClassifier_{}.dat".format(version)
        # ),
        # LinearSVCSketchClassifier(
        #     filename="../data/LinearSVCSketchClassifier_{}.dat".format(version)
        # )
    ]

    evaluate_classifiers(classifiers)


