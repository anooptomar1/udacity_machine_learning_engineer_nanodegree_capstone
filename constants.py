import os

def get_full_file_path(file_path):
    root_folder = os.path.dirname(os.path.realpath(__file__))
    return "{}/{}".format(root_folder, file_path)

DEFAULT_IMAGE_ROOT = get_full_file_path("../png")

DEFAULT_PROCESSED_IMAGE_ROOT = get_full_file_path("../processed_png")

DEFAULT_DATA_ROOT = get_full_file_path("data/")

DEFAULT_RESULTS_ROOT = get_full_file_path("results/")

#TRAIN_FEATURES_FILENAME = get_full_file_path("data/training_image_feature_vectors.npy")

TRAINED_PARAMS_FILENAME = get_full_file_path("data/params_clusters-400_winratio-0.125_winoverlap-2.5_trainingsize-8136.dat")
TRAINED_COOKBOOK_FILENAME = get_full_file_path("data/cookbook_clusters-400_winratio-0.125_winoverlap-2.5_trainingsize-8136.dat")
TRAINED_CODELABELS_FILENAME = get_full_file_path("data/codelabels_clusters-400_winratio-0.125_winoverlap-2.5_trainingsize-8136.npy")
TRAINED_CLASSIFIER_FILENAME = get_full_file_path("data/classifier-best_clusters-400_winratio-0.125_winoverlap-2.5_trainingsize-8136.dat")

WEB_PARAMS_FILENAME = get_full_file_path("data/web_params_clusters-400_winratio-0.125_winoverlap-2.5_trainingsize-11900.dat")
WEB_COOKBOOK_FILENAME = get_full_file_path("data/web_cookbook_clusters-400_winratio-0.125_winoverlap-2.5_trainingsize-11900.dat")
WEB_CODELABELS_FILENAME = get_full_file_path("data/web_codelabels_clusters-400_winratio-0.125_winoverlap-2.5_trainingsize-11900.npy")
WEB_CLASSIFIER_FILENAME = get_full_file_path("data/web_classifier-best_clusters-400_winratio-0.125_winoverlap-2.5_trainingsize-11900.dat")


SUBSET_LABELS_FILENAME = get_full_file_path("subset_labels.csv")
SUBSETB_LABELS_FILENAME = get_full_file_path("subset_b_labels.csv")

TEST_SET_FILENAME = get_full_file_path("test_set.json")
TEST_SETB_FILENAME = get_full_file_path("test_set_b.json")

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