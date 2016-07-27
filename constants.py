import os

def get_full_file_path(file_path):
    root_folder = os.path.dirname(os.path.realpath(__file__))
    return "{}/{}".format(root_folder, file_path)

DEFAULT_IMAGE_ROOT = get_full_file_path("../png")

DEFAULT_PROCESSED_IMAGE_ROOT = get_full_file_path("../processed")

DEFAULT_DATA_ROOT = get_full_file_path("data/")

DEFAULT_RESULTS_ROOT = get_full_file_path("results/")

TRAIN_FEATURES_FILENAME = get_full_file_path("data/training_image_feature_vectors.npy")

TRAIN_COOKBOOK_FILENAME = get_full_file_path("data/train_codebook_v1.dat")

TRAIN_CODE_LABELS_FILENAME = get_full_file_path("data/train_image_codelabels_v1.npy")

TRAIN_LABELS_FILENAME = get_full_file_path("data/train_image_labels_v1.npy")


class ClassifierKeys(object):
    SVM = "svm"
    LinearSVM = "linear_svm"
    KNN = "knn"
    MultinomialNaiveBayes = "multinomial_nb"
    GaussianNaiveBayes = "gaussian_nb"
    Best = "best"


class ParamKeys(object):
    NUM_CLASSES = "num_classes"  # aka labels
    WINDOW_RATIO = "window_ratio"
    WINDOW_OVERLAP = "window_overlap"
    NUM_CLUSTERS = "num_clusters"
    CLASSIFIER = "classifier"
    TRAINING_SIZE = "training_size"
    TEST_SIZE = "test_size"
    IMAGE_SIZE = "image_size"