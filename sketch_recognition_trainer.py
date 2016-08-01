import datetime
import multiprocessing
import os
#import pickle
import cPickle as pickle
import gzip
import time
import json

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from feature_extractor import SiftFeatureExtractor
import sketch_classifier
from constants import *


class BaseSketchRecognition(object):
    """
    Base class for wrapping the process of sketch recognition (concrete classes created for
    training and classifying)
    """

    def __init__(self):
        self._start_time = 0
        self._end_time = 0

        self.run_parallel_processors = True

        self.feature_extractor = None

        self.clustering = None

        self.codebook = None

        self.classifier = None

    @property
    def is_cookbook_available(self):
        """
        :return: True if codebook has been created and is loaded
        """
        return self.codebook is not None

    def start_timer(self):
        self._start_time = time.time()

    def stop_timer(self):
        self._end_time = time.time()
        et = self._end_time - self._start_time
        return et * 1000.0

    @property
    def elapsed_time(self):
        et = self._end_time - self._start_time
        return et * 1000.0

    def extract_image_descriptors_worker_for_images(self, task_queue, result_queue):
        """
        an individual worker for parallel processing
        :param task_queue:
        :param result_queue:
        :return:
        """
        while True:
            next_task = task_queue.get()
            if next_task is None:
                task_queue.task_done()
                break

            idx = next_task[0]
            image = next_task[1]
            descriptors = self.feature_extractor.compute(image)

            task_queue.task_done()
            result_queue.put((idx, descriptors))

        return

    def extract_image_descriptors(self, images, num_workers=multiprocessing.cpu_count() * 2):
        """
        responsible for extracting descriptors from the list of images, if num_workers is > 1
        then the task will be parallelized otherwise
        :param images:
        :param num_workers:
        :return:
        """

        parallel_processor_enabled = self.run_parallel_processors and num_workers > 1 and len(images) > 5

        if parallel_processor_enabled:
            print "extracting features/descriptors from {} images with {} processes".format(len(images), num_workers)
        else:
            print "extracting features/descriptors from {} images".format(len(images))

        start_time = time.time()

        if parallel_processor_enabled:
            tasks = multiprocessing.JoinableQueue()
            results = multiprocessing.Queue()

            workers = [multiprocessing.Process(
                target=self.extract_image_descriptors_worker_for_images,
                args=(tasks, results,)) for _ in range(num_workers)]

            for worker in workers:
                worker.start()

            num_jobs = len(images)

            for idx in range(num_jobs):
                tasks.put((idx, images[idx]))

            # Add a poison pill for each worker
            for i in xrange(num_workers):
                tasks.put(None)

            # Wait for all of the tasks to finish
            tasks.join()

            image_descriptors_tuples = []

            # Get the results
            while num_jobs:
                result = results.get()
                if result is not None:
                    image_descriptors_tuples.append(result)
                num_jobs -= 1

            image_descriptors_tuples = sorted(image_descriptors_tuples, key=lambda item: item[0])

            image_descriptors = [item[1] for item in image_descriptors_tuples]
        else:
            image_descriptors = self.feature_extractor.compute(images[0])

            for idx in range(1, len(images)):
                image = images[idx]
                descriptors = self.feature_extractor.compute(image)
                image_descriptors.append(descriptors)

        end_time = time.time()
        et = end_time - start_time

        print 'function extract_image_descriptors took %0.3f ms' % (et * 1000.0)

        return np.array(image_descriptors)

    def stack_image_descriptors(self, image_descriptors):
        print "stack_image_descriptors {}".format(image_descriptors.shape)

        descriptors = image_descriptors[0]
        for i in range(1, image_descriptors.shape[0]):
            descriptors = np.vstack((descriptors, image_descriptors[i]))

        return descriptors

    def code_labels_for_images(self, images):
        """
        extract descriptors from the given images then extract and return code labels for each image
        (requires codebook to have been created)
        :param images:
        :return:
        """
        image_descriptors = self.extract_image_descriptors(images=images)
        return self.code_labels_for_image_descriptors(image_descriptors)

    def code_labels_for_image_descriptors(self, image_descriptors):
        """
        extract and return the codelabels (visual words) for a given images descriptors
        :param image_descriptors:
        :return:
        """
        if self.clustering is None or self.codebook is None:
            raise "In invalid state, require kmeans and codebook to be initilised"

        images_code_labels = None

        for descriptor in image_descriptors:
            image_code_labels = self.code_labels_for_image_descriptor(descriptor)
            if images_code_labels is None:
                images_code_labels = image_code_labels
            else:
                images_code_labels = np.vstack((images_code_labels, image_code_labels))

        return images_code_labels

    def code_labels_for_image_descriptor(self, image_descriptor):
        """
        return the code label (visual word) for a given image descriptor (uses the clustering
        model, self.clustering, to predict the cluster (code label) for a given descriptor)
        :param image_descriptor:
        :return:
        """
        if self.clustering is None or self.codebook is None:
            raise "In invalid state, require kmeans and codebook to be initilised"

        labels = self.clustering.predict(image_descriptor)

        feature_vector = np.zeros(self.params['num_clusters'])

        for i, item in enumerate(image_descriptor):
            feature_vector[labels[i]] += 1

        code_labels = np.array(feature_vector, np.float)
        code_labels = self.normalize(code_labels)

        return code_labels

    def normalize(self, data):
        """
        remove the mean from the given vector
        :param data:
        :return:
        """
        sum_input = np.sum(data)
        if sum_input > 0:
            return data / sum_input
        else:
            return data

    def load_clustering_and_cookbook_from_file(self, filename):
        """
        load, if exists, the clustering model and cookbook
        :param filename:
        :return:
        """
        if not os.path.isfile(filename):
            return False

        with gzip.GzipFile(filename, 'r') as f:
            self.clustering, self.codebook = pickle.load(f)

        return True

    def save_clustering_and_cookbook_to_file(self, filename):
        """
        persist the clustering model and codebook
        :param filename:
        :return:
        """
        with gzip.GzipFile(filename, 'w') as f:
            pickle.dump((self.clustering, self.codebook), f)

    def load_classifier_from_file(self, filename):
        """
        :param filename:
        :return:
        """
        if not os.path.isfile(filename):
            return False

        with gzip.GzipFile(filename, 'r') as f:
            self.classifier = sketch_classifier.BaseSketchClassifier(filename=filename)

        return True

    def save_classifier_to_file(self, filename):
        """
        :param filename:
        :return:
        """

        if self.classifier is None:
            return False

        self.classifier.store(filename=filename)

        return True


class SketchRecognitionTrainer(BaseSketchRecognition):
    """
    Wrap the process of creating the codebook for a given set of images and their labels
    """

    def __init__(self, file_path, run_parallel_processors=True, params=None):
        BaseSketchRecognition.__init__(self)

        self.file_path = file_path

        self.run_parallel_processors = run_parallel_processors

        self.timings = {}

        self.params = None

        self.num_clusters = 400  # number of code labels (visual words/vocabulary)

        self.restore_from_file()

        self.set_params(params)

    def set_params(self, params):
        self.params = params

        self.feature_extractor = self.create_feature_extractor()

        if ParamKeys.NUM_CLUSTERS in self.params:
            self.num_clusters = self.params[ParamKeys.NUM_CLUSTERS]

        if ParamKeys.WINDOW_RATIO in self.params:
            if self.feature_extractor is not None:
                self.feature_extractor.window_ratio = self.params[ParamKeys.WINDOW_RATIO]

    def create_feature_extractor(self):
        return SiftFeatureExtractor(self.params)

    def restore_from_file(self):
        if self.file_path is None:
            return

        self.load_clustering_and_cookbook_from_file(self.file_path)

        print "loaded kmeans model and codebook to {}".format(self.file_path)

    def persist_to_file(self):
        if self.file_path is None:
            return

        self.save_clustering_and_cookbook_to_file(self.file_path)

        print "saved kmeans model and codebook to {}".format(self.file_path)

    def create_codebook_from_images(self, images, num_retries=3):
        """
        extract the descriptors from the list of images and feeds these into the
        self.create_codebook_from_image_descriptors method to create the codebook
        :param images:
        :param num_retries:
        :return:
        """
        image_descriptors = self.extract_image_descriptors(images=images)
        return self.create_codebook_from_image_descriptors(image_descriptors=image_descriptors, num_retries=num_retries)

    def create_codebook_from_image_descriptors(self, image_descriptors, num_retries=8):
        """
        creates the codebook from a given images descriptors - this essentially means
        fitting a clustering model with the descriptors and persisting the centroids as the vocabulary
        that will be used to describe the images (or more specifically, build a histogram that describes
        a image)
        :param image_descriptors:
        :param num_retries:
        :return:
        """
        print "Begining create_codebook_from_image_descriptors"

        start_time = time.time()

        descriptors = self.stack_image_descriptors(image_descriptors=image_descriptors)

        print 'getting descriptors took %0.3f ms' % ((time.time() - start_time) * 1000.0)

        num_clusters = self.num_clusters

        print "creating codebook from {} image descriptors with {} clusters".format(
            descriptors.shape,
            num_clusters
        )

        if self.clustering is None:
            """
            if a clustering
            """
            self.clustering = MiniBatchKMeans(
                init='k-means++',
                n_clusters=num_clusters,
                batch_size=100,
                max_no_improvement=10,
                verbose=0
            )

        print "fitting {}".format(self.clustering)

        res = self.clustering.fit(descriptors)
        self.codebook = res.cluster_centers_

        self.persist_to_file()

        end_time = time.time()
        et = end_time - start_time
        print 'function create_codebook_from_image_descriptors took %0.3f ms' % (et * 1000.0)

        return self.codebook, self.clustering

    def train_and_build_classifier(self, train_labels, train_images):
        """
        wraps the process of creating the codebook and training the classifier
        :param train_labels:
        :param train_images:
        :return:
        """

        print("=== Training ===")

        print("0. Savings params")
        with open(SketchRecognitionTrainer.get_params_filename_for_params(params=self.params), 'w') as f:
            json.dump(self.params, f)

        # Image Features from training set
        if not os.path.isfile(SketchRecognitionTrainer.get_training_features_filename_for_params(params=self.params)):
            print("1. Extracting image descriptors")

            self.start_timer()
            train_image_features = self.extract_image_descriptors(train_images)
            self.timings["1_extracting_descriptors"] = self.stop_timer()
            np.save(
                SketchRecognitionTrainer.get_training_features_filename_for_params(params=self.params),
                train_image_features
            )

            print "... extracted and saved features extracted from images"
        else:
            print("1. Loading image descriptors")
            train_image_features = np.load(
                SketchRecognitionTrainer.get_training_features_filename_for_params(params=self.params)
            )
            print "... loaded features from file"

        # Codebook
        if self.codebook is None:
            print("2. Creating codebook")

            self.start_timer()
            self.create_codebook_from_image_descriptors(image_descriptors=train_image_features)
            self.timings["2_creating_codebook"] = self.stop_timer()

        # Codelabels for training set
        if not os.path.isfile(SketchRecognitionTrainer.get_training_codelabels_filename_for_params(params=self.params)):
            print("3. Creating codelabels for training images")

            self.start_timer()
            train_images_codelabels = self.code_labels_for_image_descriptors(train_image_features)
            self.timings["3_creating_codelabels_for_training_data"] = self.stop_timer()
            np.save(
                SketchRecognitionTrainer.get_training_codelabels_filename_for_params(params=self.params),
                train_images_codelabels
            )
            print("...saved Code labels for training set image features")
        else:
            print("3. Loading codelabels for training images")

            train_images_codelabels = np.load(
                SketchRecognitionTrainer.get_training_codelabels_filename_for_params(params=self.params)
            )
            print "... loaded Code labels for training set image features"

        # training classifier
        if self.classifier is None:
            self.classifier = self.build_default_classifier()

        if not self.classifier.is_trained:
            print("4. Training classifier")
            self.start_timer()
            self.classifier.fit(X=train_images_codelabels, labels=train_labels)
            self.timings["4_training_classifier"] = self.stop_timer()

        return self.classifier

    def build_default_classifier(self):
        return sketch_classifier.SketchClassifierFactory.create_classifier(
            self.params[ParamKeys.CLASSIFIER],
            filename=SketchRecognitionTrainer.get_classifier_filename_for_params(params=self.params)
        )

    @staticmethod
    def get_params_filename_for_params(root=DEFAULT_DATA_ROOT, params=None):

        filename = "params_clusters-{}_winratio-{}_winoverlap-{}_trainingsize-{}".format(
            params[ParamKeys.NUM_CLUSTERS],
            params[ParamKeys.WINDOW_RATIO],
            params[ParamKeys.WINDOW_OVERLAP],
            params[ParamKeys.TRAINING_SIZE]
        )

        if "prefix" in params:
            filename = "{}_{}".format(params["prefix"], filename)

        if "postfix" in params:
            filename = "{}_{}".format(params["postfix"], filename)

        return "{}{}.dat".format(root, filename)

    @staticmethod
    def get_cookbook_filename_for_params(root=DEFAULT_DATA_ROOT, params=None):

        filename = "cookbook_clusters-{}_winratio-{}_winoverlap-{}_trainingsize-{}".format(
            params[ParamKeys.NUM_CLUSTERS],
            params[ParamKeys.WINDOW_RATIO],
            params[ParamKeys.WINDOW_OVERLAP],
            params[ParamKeys.TRAINING_SIZE]
        )

        if "prefix" in params:
            filename = "{}_{}".format(params["prefix"], filename)

        if "postfix" in params:
            filename = "{}_{}".format(params["postfix"], filename)

        return "{}{}.dat".format(root, filename)

    @staticmethod
    def get_training_features_filename_for_params(root=DEFAULT_DATA_ROOT, params=None):

        filename = "trainingfeatures_clusters-{}_winratio-{}_winoverlap-{}_trainingsize-{}".format(
            params[ParamKeys.NUM_CLUSTERS],
            params[ParamKeys.WINDOW_RATIO],
            params[ParamKeys.WINDOW_OVERLAP],
            params[ParamKeys.TRAINING_SIZE]
        )

        if "prefix" in params:
            filename = "{}_{}".format(params["prefix"], filename)

        if "postfix" in params:
            filename = "{}_{}".format(params["postfix"], filename)

        return "{}{}.npy".format(root, filename)

    @staticmethod
    def get_training_codelabels_filename_for_params(root=DEFAULT_DATA_ROOT, params=None):

        filename = "codelabels_clusters-{}_winratio-{}_winoverlap-{}_trainingsize-{}".format(
            params[ParamKeys.NUM_CLUSTERS],
            params[ParamKeys.WINDOW_RATIO],
            params[ParamKeys.WINDOW_OVERLAP],
            params[ParamKeys.TRAINING_SIZE]
        )

        if "prefix" in params:
            filename = "{}_{}".format(params["prefix"], filename)

        if "postfix" in params:
            filename = "{}_{}".format(params["postfix"], filename)

        return "{}{}.npy".format(root, filename)

    @staticmethod
    def get_classifier_filename_for_params(root=DEFAULT_DATA_ROOT, params=None):
        filename = "classifier-{}_clusters-{}_winratio-{}_winoverlap-{}_trainingsize-{}".format(
            params[ParamKeys.CLASSIFIER],
            params[ParamKeys.NUM_CLUSTERS],
            params[ParamKeys.WINDOW_RATIO],
            params[ParamKeys.WINDOW_OVERLAP],
            params[ParamKeys.TRAINING_SIZE]
        )

        if "prefix" in params:
            filename = "{}_{}".format(params["prefix"], filename)

        if "postfix" in params:
            filename = "{}_{}".format(params["postfix"], filename)

        return "{}{}.dat".format(root, filename)

    @staticmethod
    def get_evaluation_filename_for_params(root=DEFAULT_RESULTS_ROOT, params=None):
        ts = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d%H%M')

        filename = "results_classifier-{}_clusters-{}_winratio-{}_winoverlap-{}_trainingsize-{}".format(
            params[ParamKeys.CLASSIFIER],
            params[ParamKeys.NUM_CLUSTERS],
            params[ParamKeys.WINDOW_RATIO],
            params[ParamKeys.WINDOW_OVERLAP],
            params[ParamKeys.TRAINING_SIZE]
        )

        if "prefix" in params:
            filename = "{}_{}".format(params["prefix"], filename)

        if "postfix" in params:
            filename = "{}_{}".format(params["postfix"], filename)

        return "{}{}.json".format(root, filename)


class MeanShiftSketchRecognitionTrainer(SketchRecognitionTrainer):

    def create_codebook_from_image_descriptors(self, image_descriptors, num_retries=8):
        from sklearn.cluster import MeanShift, estimate_bandwidth

        print "Begining create_codebook_from_image_descriptors"

        start_time = time.time()

        descriptors = self.stack_image_descriptors(image_descriptors=image_descriptors)

        # The following bandwidth can be automatically detected using
        bandwidth = estimate_bandwidth(descriptors, quantile=0.2, n_samples=500)

        print 'getting descriptors took %0.3f ms' % ((time.time() - start_time) * 1000.0)

        self.clustering = MeanShift(bandwidth=bandwidth, bin_seeding=True)

        res = self.clustering.fit(descriptors)
        self.codebook = res.cluster_centers_

        self.persist_to_file()

        end_time = time.time()
        et = end_time - start_time
        print 'function create_codebook_from_image_descriptors took %0.3f ms' % (et * 1000.0)

        return self.codebook, self.clustering


class SketchRecognitionClassifier(BaseSketchRecognition):

    def __init__(self, params_filename, cookbook_filename, classifier_filename):
        BaseSketchRecognition.__init__(self)

        self.params = {}

        if params_filename is None or cookbook_filename is None:
            raise Exception('Instance requires params_filename, cookbook_filename')

        if not self.load_params(filename=params_filename):
            raise Exception('Instance requires valid params file')

        if not self.load_clustering_and_cookbook_from_file(filename=cookbook_filename):
            raise Exception('Instance requires valid cookbook file')

        if classifier_filename is not None:
            if not self.load_classifier_from_file(filename=classifier_filename):
                raise Exception('Instance requires valid classifier file')

        self.feature_extractor = self.create_feature_extractor()

    def load_params(self, filename):

        if not os.path.isfile(filename):
            return False

        with open(filename, 'r') as f:
            loaded_params = json.load(f)
            self.params = loaded_params

        return True

    def get_train_labels(self):
        if self.classifier is None:
            return None

        if self.classifier.label_encoder is None:
            return None

        return self.classifier.label_encoder.classes_

    def create_feature_extractor(self):
        return SiftFeatureExtractor(self.params)

    def predict(self, images):
        if not self.in_valid_state:
            raise Exception("SketchRecognitionClassifier in invalid state")

        code_labels = self.code_labels_for_image_descriptors(
            self.extract_image_descriptors(images=images)
        )

        prediction_probs = self.classifier.predict_proba(code_labels)

        print "TODO finish prediction method"
        print prediction_probs

    @property
    def in_valid_state(self):
        return self.classifier is not None and self.clustering is not None and self.codebook is not None






