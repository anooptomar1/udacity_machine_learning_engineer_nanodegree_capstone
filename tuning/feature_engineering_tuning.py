import numpy as np

from evaluator import Evaluator
from sketch_recognition_trainer import MeanShiftSketchRecognitionTrainer
from sketch_recognition_trainer import SketchRecognitionTrainer
from tuning_helper import *


def baseline_test():
    """ baseline test, all all parameters from experimentation """

    train_labels, train_images, test_labels, test_images = get_training_and_test_data()

    selected_labels = list(set(train_labels))

    params = build_params(num_classes=len(selected_labels),
                          training_size=len(train_images),
                          test_size=len(test_images))

    trainer = SketchRecognitionTrainer(
        file_path=SketchRecognitionTrainer.get_cookbook_filename_for_params(params=params),
        run_parallel_processors=True,
        params=params
    )

    classifier = trainer.train_and_build_classifier(train_labels, train_images)
    encoded_test_labels = classifier.le.transform(test_labels)

    test_images_codelabels = trainer.code_labels_for_image_descriptors(
        trainer.extract_image_descriptors(test_images)
    )

    evaluator = Evaluator(
        clf=classifier.clf,
        label_encoder=classifier.le,
        params=params,
        output_filepath=SketchRecognitionTrainer.get_evaluation_filename_for_params(params=params)
    )

    # add timings to output
    evaluator.results["timings"] = {}
    for key, value in trainer.timings.iteritems():
        evaluator.results["timings"][key] = value

    # add comment
    evaluator.results["desc"] = "After many iterations, this is a baseline for which tuning will benchmark from"

    evaluation_results = evaluator.evaluate(X=test_images_codelabels, y=encoded_test_labels)
    print evaluation_results


def sanity_check():
    """ baseline test, all all parameters from experimentation """

    train_labels, train_images, test_labels, test_images = get_training_and_test_data()

    train_labels, train_images = get_subset_of_training_data(train_labels, train_images, split=0.05)

    selected_labels = list(set(train_labels))

    params = build_params(
        num_classes=len(selected_labels),
        training_size=len(train_images),
        test_size=len(test_images),
        fn_prefix="sanitycheck"
    )

    trainer = SketchRecognitionTrainer(
        file_path=SketchRecognitionTrainer.get_cookbook_filename_for_params(params=params),
        run_parallel_processors=True,
        params=params
    )

    classifier = trainer.train_and_build_classifier(train_labels, train_images)
    encoded_test_labels = classifier.le.transform(test_labels)

    test_images_codelabels = trainer.code_labels_for_image_descriptors(
        trainer.extract_image_descriptors(test_images)
    )

    evaluator = Evaluator(
        clf=classifier.clf,
        label_encoder=classifier.le,
        params=params,
        output_filepath=SketchRecognitionTrainer.get_evaluation_filename_for_params(params=params)
    )

    # add timings to output
    evaluator.results["timings"] = {}
    for key, value in trainer.timings.iteritems():
        evaluator.results["timings"][key] = value

    # add comment
    evaluator.results["desc"] = "After many iterations, this is a baseline for which tuning will benchmark from"

    evaluation_results = evaluator.evaluate(X=test_images_codelabels, y=encoded_test_labels)
    print evaluation_results


def training_size_test(split=0.5):
    """
    see what effect the training size has on the performance, initially take off 50%
    """
    print "test_2"

    train_labels, train_images, test_labels, test_images = get_training_and_test_data()

    train_labels, train_images = get_subset_of_training_data(train_labels, train_images, split=split)

    selected_labels = list(set(train_labels))

    params = build_params(num_classes=len(selected_labels),
                          training_size=len(train_images),
                          test_size=len(test_images))

    trainer = SketchRecognitionTrainer(
        file_path=SketchRecognitionTrainer.get_cookbook_filename_for_params(params=params),
        run_parallel_processors=True,
        params=params
    )

    classifier = trainer.train_and_build_classifier(train_labels, train_images)
    encoded_test_labels = classifier.le.transform(test_labels)

    test_images_codelabels = trainer.code_labels_for_image_descriptors(
        trainer.extract_image_descriptors(test_images)
    )

    evaluator = Evaluator(
        clf=classifier.clf,
        label_encoder=classifier.le,
        params=params,
        output_filepath=SketchRecognitionTrainer.get_evaluation_filename_for_params(params=params)
    )

    # add timings to output
    evaluator.results["timings"] = {}
    for key, value in trainer.timings.iteritems():
        evaluator.results["timings"][key] = value

    # add comment
    evaluator.results["desc"] = "After many iterations, this is a baseline for which tuning will benchmark from"

    evaluation_results = evaluator.evaluate(X=test_images_codelabels, y=encoded_test_labels)
    print evaluation_results


def cluster_size_test(num_clusters=200):
    """  """

    train_labels, train_images, test_labels, test_images = get_training_and_test_data()

    selected_labels = list(set(train_labels))

    params = build_params(num_classes=len(selected_labels),
                          training_size=len(train_images),
                          test_size=len(test_images),
                          num_clusters=num_clusters)

    trainer = SketchRecognitionTrainer(
        file_path=SketchRecognitionTrainer.get_cookbook_filename_for_params(params=params),
        run_parallel_processors=True,
        params=params
    )

    classifier = trainer.train_and_build_classifier(train_labels, train_images)
    encoded_test_labels = classifier.le.transform(test_labels)

    test_images_codelabels = trainer.code_labels_for_image_descriptors(
        trainer.extract_image_descriptors(test_images)
    )

    evaluator = Evaluator(
        clf=classifier.clf,
        label_encoder=classifier.le,
        params=params,
        output_filepath=SketchRecognitionTrainer.get_evaluation_filename_for_params(params=params)
    )

    # add timings to output
    evaluator.results["timings"] = {}
    for key, value in trainer.timings.iteritems():
        evaluator.results["timings"][key] = value

    # add comment
    evaluator.results["desc"] = "testing influence of num_clusters, set to {}".format(num_clusters)

    evaluation_results = evaluator.evaluate(X=test_images_codelabels, y=encoded_test_labels)
    print evaluation_results


def window_resolution_test(window_resolution=0.125):
    """  """

    train_labels, train_images, test_labels, test_images = get_training_and_test_data()

    # split to make experimentation easier
    train_labels, train_images = get_subset_of_training_data(train_labels, train_images, split=0.5)

    training_size = len(train_labels)

    selected_labels = list(set(train_labels))

    params = build_params(num_classes=len(selected_labels),
                          training_size=len(train_images),
                          test_size=len(test_images),
                          window_resolution=window_resolution)

    trainer = SketchRecognitionTrainer(
        file_path=SketchRecognitionTrainer.get_cookbook_filename_for_params(params=params),
        run_parallel_processors=True,
        params=params
    )

    classifier = trainer.train_and_build_classifier(train_labels, train_images)
    encoded_test_labels = classifier.le.transform(test_labels)

    test_images_codelabels = trainer.code_labels_for_image_descriptors(
        trainer.extract_image_descriptors(test_images)
    )

    evaluator = Evaluator(
        clf=classifier.clf,
        label_encoder=classifier.le,
        params=params,
        output_filepath=SketchRecognitionTrainer.get_evaluation_filename_for_params(params=params)
    )

    # add timings to output
    evaluator.results["timings"] = {}
    for key, value in trainer.timings.iteritems():
        evaluator.results["timings"][key] = value

    # add comment
    evaluator.results["desc"] = "testing influence of window_resolution, set to {}. NB training size = {}".format(
        window_resolution,
        training_size
    )

    evaluation_results = evaluator.evaluate(X=test_images_codelabels, y=encoded_test_labels)
    print evaluation_results


def window_overlap_test(window_overlap=2.):
    """  """

    train_labels, train_images, test_labels, test_images = get_training_and_test_data()

    # split to make experimentation quicker
    train_labels, train_images = get_subset_of_training_data(train_labels, train_images, split=0.5)

    training_size = len(train_labels)

    desc = "testing influence of window_overlap, set to {}. NB training size = {}".format(
        window_overlap,
        training_size
    )

    print desc

    selected_labels = list(set(train_labels))

    params = build_params(num_classes=len(selected_labels),
                          training_size=len(train_images),
                          test_size=len(test_images),
                          window_overlap=window_overlap,
                          fn_prefix="winoverlap-{}".format(window_overlap))

    trainer = SketchRecognitionTrainer(
        file_path=SketchRecognitionTrainer.get_cookbook_filename_for_params(params=params),
        run_parallel_processors=True,
        params=params
    )

    classifier = trainer.train_and_build_classifier(train_labels, train_images)
    encoded_test_labels = classifier.le.transform(test_labels)

    test_images_codelabels = trainer.code_labels_for_image_descriptors(
        trainer.extract_image_descriptors(test_images)
    )

    evaluator = Evaluator(
        clf=classifier.clf,
        label_encoder=classifier.le,
        params=params,
        output_filepath=SketchRecognitionTrainer.get_evaluation_filename_for_params(params=params)
    )

    # add timings to output
    evaluator.results["timings"] = {}
    for key, value in trainer.timings.iteritems():
        evaluator.results["timings"][key] = value

    # add comment
    evaluator.results["desc"] = desc

    evaluation_results = evaluator.evaluate(X=test_images_codelabels, y=encoded_test_labels)
    print evaluation_results


def clustering_algorithm_test(clustering='kmeans'):
    """  """

    from sklearn.cluster import KMeans
    from sklearn.cluster import MiniBatchKMeans
    import multiprocessing

    train_labels, train_images, test_labels, test_images = get_training_and_test_data()

    # split to make experimentation quicker
    train_labels, train_images = get_subset_of_training_data(train_labels, train_images, split=0.5)

    training_size = len(train_labels)

    desc = "testing influence of different clustering algorithms, using {} for a training size of {}".format(
        clustering,
        training_size
    )

    print desc

    selected_labels = list(set(train_labels))

    params = build_params(num_classes=len(selected_labels),
                          training_size=len(train_images),
                          test_size=len(test_images),
                          fn_prefix="clustering-{}".format(clustering))

    trainer = SketchRecognitionTrainer(
        file_path=SketchRecognitionTrainer.get_cookbook_filename_for_params(params=params),
        run_parallel_processors=True,
        params=params
    )

    if clustering == "kmeans":
        trainer.clustering = KMeans(
            init='k-means++',
            n_clusters=params[ParamKeys.NUM_CLUSTERS],
            n_init=10,
            max_iter=10,
            tol=1.0,
            n_jobs=multiprocessing.cpu_count() if trainer.run_parallel_processors else 1
        )
    elif clustering == "minibatchkmeans":
        trainer.clustering = MiniBatchKMeans(
            init='k-means++',
            n_clusters=params[ParamKeys.NUM_CLUSTERS],
            batch_size=100,
            n_init=10,
            max_no_improvement=10,
            verbose=0
        )
    elif clustering == "meanshift":
        trainer = MeanShiftSketchRecognitionTrainer(
            file_path=SketchRecognitionTrainer.get_cookbook_filename_for_params(params=params),
            run_parallel_processors=True,
            params=params
        )


    classifier = trainer.train_and_build_classifier(train_labels, train_images)
    encoded_test_labels = classifier.le.transform(test_labels)

    test_images_codelabels = trainer.code_labels_for_image_descriptors(
        trainer.extract_image_descriptors(test_images)
    )

    evaluator = Evaluator(
        clf=classifier.clf,
        label_encoder=classifier.le,
        params=params,
        output_filepath=SketchRecognitionTrainer.get_evaluation_filename_for_params(params=params)
    )

    # add timings to output
    evaluator.results["timings"] = {}
    for key, value in trainer.timings.iteritems():
        evaluator.results["timings"][key] = value

    # add comment
    evaluator.results["desc"] = desc

    evaluation_results = evaluator.evaluate(X=test_images_codelabels, y=encoded_test_labels)
    print evaluation_results


def feature_extractor_test(feature_extractor=FeatureExtractorKeys.SIFT):
    """  """

    train_labels, train_images, test_labels, test_images = get_training_and_test_data()

    # split to make experimentation quicker
    train_labels, train_images = get_subset_of_training_data(train_labels, train_images, split=0.8)

    training_size = len(train_labels)

    desc = "testing influence of feature_extractor, set to {}. NB training size = {}".format(
        feature_extractor,
        training_size
    )

    selected_labels = list(set(train_labels))

    params = build_params(num_classes=len(selected_labels),
                          training_size=len(train_images),
                          test_size=len(test_images),
                          feature_extractor=feature_extractor,
                          fn_prefix="descriptor")

    trainer = SketchRecognitionTrainer(
        file_path=SketchRecognitionTrainer.get_cookbook_filename_for_params(params=params),
        run_parallel_processors=True,
        params=params
    )

    classifier = trainer.train_and_build_classifier(train_labels, train_images)
    encoded_test_labels = classifier.le.transform(test_labels)

    test_images_codelabels = trainer.code_labels_for_image_descriptors(
        trainer.extract_image_descriptors(test_images)
    )

    evaluator = Evaluator(
        clf=classifier.clf,
        label_encoder=classifier.le,
        params=params,
        output_filepath=SketchRecognitionTrainer.get_evaluation_filename_for_params(params=params)
    )

    # add timings to output
    evaluator.results["timings"] = {}
    for key, value in trainer.timings.iteritems():
        evaluator.results["timings"][key] = value

    # add comment
    evaluator.results["desc"] = desc

    evaluation_results = evaluator.evaluate(X=test_images_codelabels, y=encoded_test_labels)
    print evaluation_results


def build_cookbook_and_featurevectors_for_model_tuning():
    train_labels, train_images, test_labels, test_images = get_training_and_test_data()

    selected_labels = list(set(train_labels))

    params = build_params(num_classes=len(selected_labels),
                          training_size=len(train_images),
                          test_size=len(test_images),
                          feature_extractor=FeatureExtractorKeys.SIFT,
                          window_resolution=0.125,
                          window_overlap=2.0,
                          num_clusters=450,
                          image_size=256)

    trainer = SketchRecognitionTrainer(
        file_path=TRAIN_COOKBOOK_FILENAME,
        run_parallel_processors=True,
        params=params
    )

    # 1 - extract image feature vectors
    if os.path.isfile(TRAIN_FEATURES_FILENAME):
        train_image_features = np.load(TRAIN_FEATURES_FILENAME)
    else:
        train_image_features = trainer.extract_image_descriptors(train_images)
        np.save(TRAIN_FEATURES_FILENAME, train_image_features)

    # 2 - create codebook from feature vectors
    if not trainer.is_cookbook_available:
        trainer.create_codebook_from_image_descriptors(image_descriptors=train_image_features)

    # 3 - create codelabels (visual word histograms) for each image
    if os.path.isfile(TRAIN_CODE_LABELS_FILENAME):
        train_images_codelabels = np.load(TRAIN_CODE_LABELS_FILENAME)
    else:
        train_images_codelabels = trainer.code_labels_for_image_descriptors(train_image_features)
        np.save(TRAIN_CODE_LABELS_FILENAME, train_images_codelabels)  # features

    # 4 - save labels
    np.save(TRAIN_LABELS_FILENAME, train_labels)

    print "finished creating and saving feature data:\n- codebook: {}\n- train image code labels: {}" \
          "\n- train image labels: {}".format(
        "data/codebook_v1.dat",
        "data/train_image_codelabels_v1.npy",
        "data/train_image_labels_v1.npy"
    )


if __name__ == '__main__':
    print __file__

    print baseline_test()

    """ Testing affects of training set size """

    for test_size in [0.3, 0.5, 0.8, 1.0]:
        print training_size_test(split=test_size)

    """ Testing affects of adjusting cluster size """
    num_clusters_list = [200, 400, 600, 800, 1000, 1200]
    for num_clusters in num_clusters_list:
        print cluster_size_test(num_clusters=num_clusters)

    """ Testing influence from adjusting window resolutions """
    window_resolutions = [0.01, 0.05, 0.08, 0.10, 0.125, 0.25, 0.30, 0.35, 0.45]
    for window_resolution in window_resolutions:
        print window_resolution_test(window_resolution=window_resolution)

    """ Testing influence from adjusting window overlap """
    window_overlaps = [1., 1.5, 2., 2.5, 3.]
    for window_overlap in window_overlaps:
        print window_overlap_test(window_overlap=window_overlap)

    """ Testing influence on clustering algorithms """
    clustering_algorithms = ['kmeans', 'minibatchkmeans', 'meanshift', 'kmeans', 'minibatchkmeans', 'meanshift']
    for clustering_algorithm in clustering_algorithms:
        print clustering_algorithm_test(clustering=clustering_algorithm)

    # feature_extractors = [
    #     FeatureExtractorKeys.SIFT,
    #     FeatureExtractorKeys.BRIEF,
    #     FeatureExtractorKeys.SKIMAGE_HOG_2
    # ]
    # for feature_extractor in feature_extractors:
    #     print feature_extractor_test(feature_extractor=feature_extractor)

    """ Based on the results of the above, lets build a codebook """
    build_cookbook_and_featurevectors_for_model_tuning()

    # sanity_check()