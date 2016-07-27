import os
import cv2
import re
import time
from constants import *


class MiscUtils(object):
    """
    A set of miscellaneous methods for handling IO and image loading and manipulation
    """

    @staticmethod
    def get_sub_directories(root_path=DEFAULT_IMAGE_ROOT):
        sub_dirs = []
        for root, dir_names, _ in os.walk("{}".format(root_path)):
            for dir_name in dir_names:
                sub_dirs.append(dir_name)

        return sub_dirs

    @staticmethod
    def load_image(file_path):
        #     input_image = cv2.imread(file_path)
        #     return cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

        return cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    @staticmethod
    def load_images(root_path=DEFAULT_IMAGE_ROOT,
                    subdir_names=[],
                    perfrom_crop_and_rescale_image=True,
                    subdir_image_limit=-1):
        """
        :param root_path: where the root directory of the images resides (expecting structure: <root>/<subdir_name>/images
        :param subdir_names: expecting dir of directory names
        :param perfrom_crop_and_rescale_image: boolean indicating if cropping and uniform rescaling is required
        :param subdir_image_limit: how many images to load from each sub-directory (-1 == all)
        :return: list of loaded images
        """

        start_time = time.time()

        images = []
        labels = []

        for subdir_name in subdir_names:
            image_counter = 0
            for root, dirnames, filenames in os.walk("{}/{}".format(root_path, subdir_name)):
                for filename in filenames:
                    if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
                        labels.append(subdir_name)

                        filepath = os.path.join("{}/{}".format(root_path, subdir_name), filename)
                        image = MiscUtils.load_image(filepath)
                        if perfrom_crop_and_rescale_image:
                            image = MiscUtils.crop_and_rescale_image(image)
                        images.append(image)
                        image_counter += 1

                        if subdir_image_limit > 0 and image_counter >= subdir_image_limit:
                            break

        end_time = time.time()
        et = end_time - start_time

        print 'function load_images took %0.3f ms' % (et * 1000.0)

        return labels, images

    @staticmethod
    def crop_and_rescale_image(image, size=256):
        """
        crop and resize image, clip to the max bounding box of the image and perform a uniform rescale
        of the image
        """

        bbox = {
            'left': -1,
            'top': -1,
            'right': -1,
            'bottom': -1,
            'width': -1,
            'height': -1
        }

        max_area = (image.shape[0] * image.shape[1]) * 0.9

        contours, hierarchy = cv2.findContours(image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        idx = 0
        for cnt in contours:
            idx += 1

            x, y, w, h = cv2.boundingRect(cnt)

            if (w * h) >= max_area:
                continue

            right = x + w
            bottom = y + h

            bbox["left"] = x if bbox["left"] == -1 else min(bbox["left"], x)
            bbox["top"] = y if bbox["top"] == -1 else min(bbox["top"], y)

            bbox["right"] = right if bbox["right"] == -1 else max(bbox["right"], right)
            bbox["bottom"] = bottom if bbox["bottom"] == -1 else max(bbox["bottom"], bottom)

        bbox['width'] = bbox['right'] - bbox['left']
        bbox['height'] = bbox['bottom'] - bbox['top']

        # scale to the largest dimension to keep the same aspect ratio
        if bbox["width"] > bbox["height"]:
            diff = (bbox["width"] - bbox["height"]) / 2
            bbox["top"] -= diff
            bbox["height"] += diff * 2
        else:
            diff = (bbox["height"] - bbox["width"]) / 2
            bbox["left"] -= diff
            bbox["width"] += diff * 2

            # clip image
        cropped = image[bbox['top']:bbox['bottom'], bbox['left']:bbox['right']]

        # re-scale image
        #scaled = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_CUBIC)
        scaled = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_NEAREST)

        return scaled

    @staticmethod
    def show_image(image, wait_key=True):
        cv2.imshow('Image', image)

        if wait_key:
            cv2.waitKey()

    @staticmethod
    def show_images(images, title="image", wait_key=True):

        for image in images:
            cv2.imshow(title, image)

            if wait_key:
                cv2.waitKey()

    @staticmethod
    def training_test_split(root_path=DEFAULT_IMAGE_ROOT, subdir_names=[], perfrom_crop_and_rescale_image=True,
                            subdir_image_limit=-1, train_test_split=0.8):
        """
        load the images and split them into a training and test set
        :param root_path:
        :param subdir_names:
        :param perfrom_crop_and_rescale_image:
        :param subdir_image_limit:
        :param train_test_split:
        :return:
        """

        start_time = time.time()

        train_labels = []
        train_images = []
        test_labels = []
        test_images = []

        for subdir_name in subdir_names:
            labels, images = MiscUtils.load_images(root_path, [subdir_name], perfrom_crop_and_rescale_image, subdir_image_limit)

            train_count = int(float(len(labels)) * float(train_test_split))
            train_labels.extend(labels[:train_count])
            train_images.extend(images[:train_count])

            test_labels.extend(labels[train_count + 1:])
            test_images.extend(images[train_count + 1:])

        end_time = time.time()
        et = end_time - start_time

        print 'function training_test_split took %0.3f ms' % (et * 1000.0)

        return train_labels, train_images, test_labels, test_images

    @staticmethod
    def crop_and_rescale_images(source_path=DEFAULT_IMAGE_ROOT, dest_path=DEFAULT_PROCESSED_IMAGE_ROOT):
        start_time = time.time()

        subdir_names = MiscUtils.get_sub_directories()[:]

        for subdir_name in subdir_names:
            labels, images = MiscUtils.load_images(source_path, [subdir_name], True, -1)

            for i in range(len(labels)):
                label = labels[i]
                img = images[i]

                directory = "{}/{}".format(dest_path, label)

                if not os.path.isdir(directory):
                    os.makedirs(directory)

                cv2.imwrite('{}/{}_{}.png'.format(directory, label, i), img)

        end_time = time.time()
        et = end_time - start_time

        print 'function crop_and_rescale_images took %0.3f ms' % (et * 1000.0)


