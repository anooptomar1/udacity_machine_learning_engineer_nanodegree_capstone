import datetime
import multiprocessing
import os
import pickle
import time

import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans

from classifiers.sketch_classifier import LinearSVCSketchClassifier
from feature_extractors.brief_feature_extractor import BriefFeatureExtractor
from feature_extractors.hog_feature_extractor import HOGFeatureExtractor
from feature_extractors.sift_feature_extractor import SiftFeatureExtractor
from feature_extractors.skimage_hog_2_feature_extractor import SKImageHOGF2eatureExtractor
from feature_extractors.skimage_hog_feature_extractor import SKImageHOGFeatureExtractor


class FeatureExtractorKeys(object):
    HOG = "hog"
    SKIMAGE_HOG = "skimage_hog"
    SKIMAGE_HOG_2 = "skimage_hog_2"
    SIFT = "sift"
    BRIEF = "breif"


class ClassifierKeys(object):
    SVM = "svm"


class ParamKeys(object):
    FEATURE_EXTRACTOR = "feature_extractor"
    FEATURE_COUNT = "feature_count"
    CELL_RESOLUTION = "cell_resolution"
    WINDOW_RATIO = "window_ratio"
    WINDOW_OVERLAP = "window_overlap"
    NUM_CLUSTERS = "num_clusters"
    CLASSIFIER = "classifier"
    TRAINING_SIZE = "training_size"
    TEST_SIZE = "test_size"
    NUM_CLASSES = "num_classes"
    IMAGE_SIZE = "image_size"


class BaseSketchRecognition(object):

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
        return self.codebook is not None

    def start_timer(self):
        self._start_time = time.time()

    def stop_timer(self):
        self._end_time = time.time()
        et = self._end_time - self._start_time
        return et * 1000.0

    def elapsed_time(self):
        et = self._end_time - self._start_time
        return et * 1000.0

    def extract_image_descriptors_worker_for_images(self, task_queue, result_queue):
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
        create worker thread to extract features from each image
        https://docs.python.org/2/library/multiprocessing.html
        https://www.quantstart.com/articles/parallelising-python-with-threading-and-multiprocessing
        https://pymotw.com/2/multiprocessing/communication.html
        :param images:
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

            # image_descriptors = image_descriptors_tuples[0][1]
            # for i in range(1, len(image_descriptors_tuples)):
            #     image_descriptors = np.vstack((image_descriptors, image_descriptors_tuples[i][1]))

            image_descriptors = [item[1] for item in image_descriptors_tuples]
        else:
            image_descriptors = self.feature_extractor.compute(images[0])

            for idx in range(1, len(images)):
                image = images[idx]
                descriptors = self.feature_extractor.compute(image)
                # image_descriptors = np.vstack((image_descriptors, descriptors))
                image_descriptors.append(descriptors)

        end_time = time.time()
        et = end_time - start_time

        print 'function extract_image_descriptors took %0.3f ms' % (et * 1000.0)

        return np.array(image_descriptors)
        # return image_descriptors

    def stack_image_descriptors(self, image_descriptors):
        print "stack_image_descriptors {}".format(image_descriptors.shape)

        descriptors = image_descriptors[0]
        for i in range(1, image_descriptors.shape[0]):
            descriptors = np.vstack((descriptors, image_descriptors[i]))

        return descriptors

    def code_labels_for_images(self, images):
        image_descriptors = self.extract_image_descriptors(images=images)
        return self.code_labels_for_image_descriptors(image_descriptors)

    def code_labels_for_image_descriptors(self, image_descriptors):
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
        if self.clustering is None or self.codebook is None:
            raise "In invalid state, require kmeans and codebook to be initilised"

        labels = self.clustering.predict(image_descriptor)

        feature_vector = np.zeros(self.num_clusters)

        for i, item in enumerate(image_descriptor):
            feature_vector[labels[i]] += 1

        # code_labels = np.reshape(feature_vector, ((1, feature_vector.shape[0])))
        code_labels = np.array(feature_vector, np.float)
        code_labels = self.normalize(code_labels)

        return code_labels

    def normalize(self, data):
        sum_input = np.sum(data)
        if sum_input > 0:
            return data / sum_input
        else:
            return data

    def load_clustering_and_cookbook_from_file(self, filename):
        if not os.path.isfile(filename):
            return False

        with open(filename, 'r') as f:
            self.clustering, self.codebook = pickle.load(f)

        return True

    def save_clustering_and_cookbook_to_file(self, filename):
        with open(filename, 'w') as f:
            pickle.dump((self.clustering, self.codebook), f)


class SketchRecognitionTrainer(BaseSketchRecognition):

    def __init__(self, file_path, run_parallel_processors=True, params=None):
        #super(SketchRecognitionTrainer, self).__init__()
        BaseSketchRecognition.__init__(self)

        self.file_path = file_path

        self.run_parallel_processors = run_parallel_processors

        self.timings = {}

        self.params = None

        self.num_clusters = 32

        self.restore_from_file()

        self.set_params(params)

    def set_params(self, params):
        self.params = params

        self.feature_extractor = self.create_feature_extractor(self.params)

        if ParamKeys.NUM_CLUSTERS in self.params:
            self.num_clusters = self.params[ParamKeys.NUM_CLUSTERS]

        if ParamKeys.WINDOW_RATIO in self.params:
            if self.feature_extractor is not None:
                self.feature_extractor.window_ratio = self.params[ParamKeys.WINDOW_RATIO]

        if ParamKeys.FEATURE_COUNT in self.params:
            if self.feature_extractor is not None:
                self.feature_extractor.feature_count = self.params[ParamKeys.FEATURE_COUNT]

    def create_feature_extractor(self, params):
        if params[ParamKeys.FEATURE_EXTRACTOR] == FeatureExtractorKeys.HOG:
            return HOGFeatureExtractor(self.params)
        elif params[ParamKeys.FEATURE_EXTRACTOR] == FeatureExtractorKeys.SKIMAGE_HOG:
            return SKImageHOGFeatureExtractor(self.params)
        elif self.params[ParamKeys.FEATURE_EXTRACTOR] == FeatureExtractorKeys.SKIMAGE_HOG_2:
            return SKImageHOGF2eatureExtractor(self.params)
        elif self.params[ParamKeys.FEATURE_EXTRACTOR] == FeatureExtractorKeys.SIFT:
            return SiftFeatureExtractor(self.params)
        elif self.params[ParamKeys.FEATURE_EXTRACTOR] == FeatureExtractorKeys.BRIEF:
            return BriefFeatureExtractor(self.params)

        raise Exception("Unknown feature extractor")

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
        image_descriptors = self.extract_image_descriptors(images=images)
        return self.create_codebook_from_image_descriptors(image_descriptors=image_descriptors, num_retries=num_retries)

    def create_codebook_from_image_descriptors(self, image_descriptors, num_retries=8):
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
            # self.clustering = KMeans(
            #     n_clusters=num_clusters,
            #     init='k-means++',
            #     n_init=num_retries,
            #     max_iter=10,
            #     tol=1.0,
            #     verbose=10,
            #     copy_x=False,
            #     n_jobs=-1 if self.run_parallel_processors else 1
            # )

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

    def train_and_build_classifier(self, train_labels, train_images,
                                   force_feature_extraction=False,
                                   force_codebook_rebuild=False,
                                   force_classifier_retraining=False):
        print "=== Training ==="

        # Image Features from training set
        if not os.path.isfile(SketchRecognitionTrainer.get_training_features_filename_for_params(params=self.params)) \
                or force_feature_extraction:
            self.start_timer()
            train_image_features = self.extract_image_descriptors(train_images)
            self.timings["1_extracting_descriptors"] = self.stop_timer()
            np.save(
                SketchRecognitionTrainer.get_training_features_filename_for_params(params=self.params),
                train_image_features
            )
            print "saved features extracted from images"
        else:
            train_image_features = np.load(
                SketchRecognitionTrainer.get_training_features_filename_for_params(params=self.params)
            )
            print "loaded features from file"

        # Codebook
        if self.codebook is None or force_codebook_rebuild:
            self.start_timer()
            self.create_codebook_from_image_descriptors(image_descriptors=train_image_features)
            self.timings["2_creating_codebook"] = self.stop_timer()

        # Codelabels for training set
        if not os.path.isfile(
                SketchRecognitionTrainer.get_training_codelabels_filename_for_params(params=self.params)):
            self.start_timer()
            train_images_codelabels = self.code_labels_for_image_descriptors(train_image_features)
            self.timings["3_creating_codelabels_for_training_data"] = self.stop_timer()
            np.save(
                SketchRecognitionTrainer.get_training_codelabels_filename_for_params(params=self.params),
                train_images_codelabels
            )
            print "saved Code labels for training set image features"
        else:
            train_images_codelabels = np.load(
                SketchRecognitionTrainer.get_training_codelabels_filename_for_params(params=self.params)
            )
            print "loaded Code labels for training set image features"

        # training classifier
        if self.classifier is None:
            self.classifier = self.build_default_classifier()

        if not self.classifier.is_trained or force_classifier_retraining:
            self.start_timer()
            self.classifier.fit(X=train_images_codelabels, labels=train_labels)
            self.timings["4_training_classifier"] = self.stop_timer()

        return self.classifier

    def build_default_classifier(self):
        return LinearSVCSketchClassifier(
            filename=SketchRecognitionTrainer.get_classifier_filename_for_params(params=self.params)
        )

    @staticmethod
    def get_cookbook_filename_for_params(root="data/", params=None):

        filename = "cookbook_extractor-{}_clusters-{}_winratio-{}_trainingsize-{}_classes-{}_features-{}".format(
            params[ParamKeys.FEATURE_EXTRACTOR],
            params[ParamKeys.NUM_CLUSTERS],
            params[ParamKeys.WINDOW_RATIO],
            params[ParamKeys.TRAINING_SIZE],
            params[ParamKeys.NUM_CLASSES],
            params[ParamKeys.FEATURE_COUNT]
        )

        if "prefix" in params:
            filename = "{}_{}".format(params["prefix"], filename)

        if "postfix" in params:
            filename = "{}_{}".format(params["postfix"], filename)

        return "{}{}.dat".format(root, filename)

    @staticmethod
    def get_training_features_filename_for_params(root="data/", params=None):
        filename = "trainingfeatures_extractor-{}_clusters-{}_winratio-{}_trainingsize-{}_classes-{}_features-{}".format(
            params[ParamKeys.FEATURE_EXTRACTOR],
            params[ParamKeys.NUM_CLUSTERS],
            params[ParamKeys.WINDOW_RATIO],
            params[ParamKeys.TRAINING_SIZE],
            params[ParamKeys.NUM_CLASSES],
            params[ParamKeys.FEATURE_COUNT]
        )

        if "prefix" in params:
            filename = "{}_{}".format(params["prefix"], filename)

        if "postfix" in params:
            filename = "{}_{}".format(params["postfix"], filename)

        return "{}{}.npy".format(root, filename)

    @staticmethod
    def get_training_codelabels_filename_for_params(root="data/", params=None):
        filename = "codelabels_extractor-{}_clusters-{}_winratio-{}_trainingsize-{}_classes-{}_features-{}.npy".format(
            params[ParamKeys.FEATURE_EXTRACTOR],
            params[ParamKeys.NUM_CLUSTERS],
            params[ParamKeys.WINDOW_RATIO],
            params[ParamKeys.TRAINING_SIZE],
            params[ParamKeys.NUM_CLASSES],
            params[ParamKeys.FEATURE_COUNT]
        )

        if "prefix" in params:
            filename = "{}_{}".format(params["prefix"], filename)

        if "postfix" in params:
            filename = "{}_{}".format(params["postfix"], filename)

        return "{}{}.npy".format(root, filename)

    @staticmethod
    def get_classifier_filename_for_params(root="data/", params=None):
        filename = "clf-{}_extractor-{}_clusters-{}_winratio-{}_trainingsize-{}_classes-{}_features-{}".format(
            params[ParamKeys.CLASSIFIER],
            params[ParamKeys.FEATURE_EXTRACTOR],
            params[ParamKeys.NUM_CLUSTERS],
            params[ParamKeys.WINDOW_RATIO],
            params[ParamKeys.TRAINING_SIZE],
            params[ParamKeys.NUM_CLASSES],
            params[ParamKeys.FEATURE_COUNT]
        )

        if "prefix" in params:
            filename = "{}_{}".format(params["prefix"], filename)

        if "postfix" in params:
            filename = "{}_{}".format(params["postfix"], filename)

        return "{}{}.json".format(root, filename)

    @staticmethod
    def get_evaluation_filename_for_params(root="results/", params=None):
        ts = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d%H%M')

        filename = "results_extractor-{}_clusters-{}_winratio-{}_trainingsize-{}_classes-{}_features-{}_{}".format(
            params[ParamKeys.FEATURE_EXTRACTOR],
            params[ParamKeys.NUM_CLUSTERS],
            params[ParamKeys.WINDOW_RATIO],
            params[ParamKeys.TRAINING_SIZE],
            params[ParamKeys.NUM_CLASSES],
            params[ParamKeys.FEATURE_COUNT],
            ts
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

    def __init__(self):
        #super(SketchRecognitionClassifier, self).__init__()
        BaseSketchRecognition.__init__(self)

        self.num_clusters = 450

        self.feature_extractor = SiftFeatureExtractor(
            {
                ParamKeys.WINDOW_RATIO: 0.125,
                ParamKeys.WINDOW_OVERLAP:2.0,
                ParamKeys.IMAGE_SIZE: 256
            }
        )

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


def load_labels(filename="subset_labels.csv"):
    labels = []
    import csv
    with open(filename, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            labels.extend(row)

    return labels

if __name__ == '__main__':
    print __file__

    ####################################################################

    # """ understanding layout of OpenCV's MAT format """
    # # shape = hxw = (1200,1600) -> image is normally wxh = (1600,1200) row,column
    # img = cv2.imread('test.png', cv2.IMREAD_GRAYSCALE)
    #
    # print type(images[0][0][0])
    #
    # print img.shape
    #
    # # red
    # print img[0,0] # same as -> print img.item((0, 0))
    # print img[1, 0]
    #
    # # green
    # print img[1, 1]
    # print img[2, 1]
    #
    # # generalised to [R,C]





