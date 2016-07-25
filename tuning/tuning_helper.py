import json
import os
import random

import misc_utils
from sketch_recognition_trainer import ClassifierKeys
from sketch_recognition_trainer import FeatureExtractorKeys
from sketch_recognition_trainer import ParamKeys

TRAIN_FEATURES_FILENAME = "../data/training_image_feature_vectors.npy"
TRAIN_COOKBOOK_FILENAME = "../data/train_codebook_v1.dat"
TRAIN_CODE_LABELS_FILENAME = "../data/train_image_codelabels_v1.npy"
TRAIN_LABELS_FILENAME = "../data/train_image_labels_v1.npy"


def build_params(num_classes, training_size, test_size,
                 feature_extractor=FeatureExtractorKeys.SIFT,
                 feature_count=324, cell_resolution=4,
                 window_resolution=0.125, window_overlap=2.0,
                 num_clusters=400,
                 classifier=ClassifierKeys.SVM,
                 image_size=256,
                 fn_prefix=None, fn_postfix=None):
    """
    TODO: describe parameters and their effect on the feature extractor
    :param num_classes:
    :param training_size:
    :param test_size:
    :param feature_extractor:
    :param feature_count:
    :param cell_resolution:
    :param window_resolution:
    :param window_overlap:
    :param num_clusters:
    :param classifier:
    :param image_size:
    :param fn_prefix:
    :param fn_postfix:
    :return:
    """

    params = dict()

    params[ParamKeys.FEATURE_EXTRACTOR] = feature_extractor
    params[ParamKeys.FEATURE_COUNT] = feature_count
    params[ParamKeys.CELL_RESOLUTION] = cell_resolution
    params[ParamKeys.WINDOW_RATIO] = window_resolution
    params[ParamKeys.WINDOW_OVERLAP] = window_overlap
    params[ParamKeys.NUM_CLUSTERS] = num_clusters
    params[ParamKeys.CLASSIFIER] = classifier
    params[ParamKeys.IMAGE_SIZE] = image_size
    params[ParamKeys.NUM_CLASSES] = num_classes
    params[ParamKeys.TRAINING_SIZE] = training_size
    params[ParamKeys.TEST_SIZE] = test_size

    if fn_prefix is not None:
        params["prefix"] = fn_prefix

    if fn_postfix is not None:
        params["postfix"] = fn_postfix

    return params


def load_labels(filename="../subset_labels.csv"):
    labels = []
    import csv
    with open(filename, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            labels.extend(row)

    return labels


def split_data(labels, images, test_set_filename="../test_set.json", test_set_size=8):
    test_details = None

    if os.path.isfile(test_set_filename):
        with open(test_set_filename, 'r') as f:
            test_details = json.load(f)

    if test_details is None:
        index_data = {}
        for i in range(len(labels)):
            label = labels[i]
            if label not in index_data:
                index_data[label] = []

            index_data[label].append(i)

        test_details = {}

        for label, indicies in index_data.iteritems():
            test_details[label] = random.sample(indicies, test_set_size)

        with open(test_set_filename, 'wb') as f:
            _results = json.dumps(test_details)
            f.write(_results)

    train_labels = []
    train_images = []
    test_labels = []
    test_images = []

    for i in range(len(labels)):
        label = labels[i]
        image = images[i]

        if i in test_details[label]:
            test_labels.append(label)
            test_images.append(image)
        else:
            train_labels.append(label)
            train_images.append(image)

    return train_labels, train_images, test_labels, test_images


def get_training_and_test_data():
    selected_labels = load_labels()

    labels, images = misc_utils.MiscUtils.load_images(root_path="../../processed_png",
                                                      subdir_names=selected_labels,
                                                      subdir_image_limit=0,
                                                      perfrom_crop_and_rescale_image=False)

    train_labels, train_images, test_labels, test_images = split_data(labels=labels, images=images)

    return train_labels, train_images, test_labels, test_images


def get_subset_of_training_data(all_train_labels, all_train_images, split=0.5):
    data_dict = {}

    if split <= 0.0 or split >= 1.0:
        return all_train_labels, all_train_images

    for i in range(len(all_train_images)):
        key = all_train_labels[i]
        img = all_train_images[i]
        if key not in data_dict:
            data_dict[key] = []

        data_dict[key].append(img)

    train_labels = []
    train_images = []

    for key, images in data_dict.iteritems():
        count = int(float(len(images)) * split)

        # add labels
        train_labels.extend([key for _ in range(count)])

        if len(images) <= count:
            train_images.extend(images)
        else:
            train_images.extend(random.sample(images, count))

    return train_labels, train_images