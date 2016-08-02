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

def rebuild_with_web_subset():
    print("=== rebuild_with_web_subset ===")

    labels = [
        'airplane',
        'alarm clock',
        'angel',
        'apple',
        'axe',
        'bee',
        'bell',
        'bicycle',
        'blimp',
        'book',
        'bulldozer',
        'bus',
        'bush',
        'butterfly',
        'cactus',
        'cake',
        'calculator',
        'camel',
        'camera',
        'candle',
        'cannon',
        'car (sedan)',
        'castle',
        'cat',
        'cell phone',
        'church',
        'cigarette',
        'cloud',
        'comb',
        'computer monitor',
        'cow',
        'crab',
        'crane (machine)',
        'crocodile',
        'crown',
        'cup',
        'diamond',
        'dog',
        'dolphin',
        'donut',
        'dragon',
        'duck',
        'elephant',
        'envelope',
        'eyeglasses',
        'face',
        'fan',
        'fish',
        'flashlight',
        'floor lamp',
        'flower with stem',
        'flying bird',
        'flying saucer',
        'fork',
        'frog',
        'giraffe',
        'guitar',
        'hammer',
        'hand',
        'head',
        'hedgehog',
        'helicopter',
        'horse',
        'hot air balloon',
        'house',
        'kangaroo',
        'key',
        'keyboard',
        'knife',
        'ladder',
        'laptop',
        'lightbulb',
        'lion',
        'mailbox',
        'megaphone',
        'mermaid',
        'monkey',
        'mouse (animal)',
        'mug',
        'mushroom',
        'octopus',
        'owl',
        'palm tree',
        'panda',
        'penguin',
        'person sitting',
        'person walking',
        'pickup truck',
        'pig',
        'pigeon',
        'pipe (for smoking)',
        'pizza',
        'potted plant',
        'present',
        'pumpkin',
        'rabbit',
        'radio',
        'rainbow',
        'revolver',
        'rifle',
        'sailboat',
        'saxophone',
        'scissors',
        'sea turtle',
        'shark',
        'sheep',
        'ship',
        'shoe',
        'shovel',
        'skateboard',
        'snail',
        'snake',
        'snowman',
        'socks',
        'space shuttle',
        'spider',
        'spoon',
        'submarine',
        'suitcase',
        'sun',
        'suv',
        'sword',
        't-shirt',
        'table',
        'tablelamp',
        'teapot',
        'teddy-bear',
        'telephone',
        'tennis-racket',
        'tent',
        'tiger',
        'tomato',
        'toothbrush',
        'tractor',
        'traffic light',
        'train',
        'tree',
        'trousers',
        'truck',
        'tv',
        'umbrella',
        'van',
        'vase',
        'violin',
        'walkie talkie',
        'windmill',
        'wine-bottle',
        'wineglass',
        'wrist-watch'
    ]

    labels, images = misc_utils.MiscUtils.load_images(root_path=DEFAULT_PROCESSED_IMAGE_ROOT,
                                                      subdir_names=labels,
                                                      subdir_image_limit=0,
                                                      perfrom_crop_and_rescale_image=False)

    train_labels, train_images, test_labels, test_images = split_data(
        labels=labels,
        images=images,
        test_set_filename="web_test.json",
        test_set_size=4)

    rebuild(train_labels, train_images, test_labels, test_images, "web")

if __name__ == '__main__':
    print __file__

    #rebuild_with_subset()
    #rebuild_with_subset_b()

    rebuild_with_web_subset()

