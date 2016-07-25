import math
import numpy as np
import time
from feature_extractor import BaseFeatureExtractor

class HOGFeatureExtractor(BaseFeatureExtractor):

    def __init__(self, params=None):
        super(HOGFeatureExtractor, self).__init__(params)

        if "feature_count" in params:
            self.feature_count = params["feature_count"]
        else:
            self.feature_count = 144

        if "cell_res" in params:
            self.cell_xres = params["cell_res"]
            self.cell_yres = params["cell_res"]
        else:
            self.cell_xres = 4
            self.cell_yres = 4

        if "cell_orient" in params:
            self.cell_orient = params["cell_orient"]
        else:
            self.cell_orient = 8

        if "window_ratio" in params:
            self.window_ratio = params["window_ratio"]
        else:
            self.window_ratio = 0.30

    def _compute(self, image):

        xres = image.shape[1]
        yres = image.shape[0]

        window_width = int(self.window_ratio * float(xres))
        window_height = int(self.window_ratio * float(yres))

        size = self.feature_count, self.cell_xres * self.cell_yres * self.cell_orient
        descriptor = np.zeros(size, dtype=np.float)

        for idx in range(self.feature_count):
            x, y = self.get_keypoint_centre(xres, yres, window_width, window_height, idx)

            orientations = self.compute_orientations_for_point(image, x, y, window_width, window_height)

            # convert orientations into a 1D descriptor
            for x in range(self.cell_xres):
                for y in range(self.cell_yres):
                    for o in range(self.cell_orient):
                        index = o + self.cell_orient * y + self.cell_orient * self.cell_yres * x
                        descriptor[idx, index] = float(orientations[x][y][o])

        """
        normalising
        http://stackoverflow.com/questions/21030391/how-to-normalize-array-numpy
        norm1 = x / np.linalg.norm(x)
        norm2 = normalize(x[:,np.newaxis], axis=0).ravel()
        """

        # normalize descriptor
        #descriptor /= np.linalg.norm(descriptor[idx], ord=1)
        for idx in range(self.feature_count):
            # norm = np.linalg.norm(descriptor[idx])
            norm = np.sum(descriptor[idx])
            if norm > 0:
                descriptor[idx] /= norm

        return descriptor

    def get_keypoint_centre(self, xres, yres, window_width, window_height, idx):

        half_window_width = window_width / 2
        half_window_height = window_height / 2

        start_x = half_window_width + 1
        end_x = xres - half_window_width - 1
        start_y = half_window_height + 1
        end_y = yres - half_window_height - 1

        x_range = end_x - start_x
        y_range = end_y - start_y
        samples_per_row = int(math.sqrt(float(self.feature_count)))
        samples_per_col = samples_per_row
        if (samples_per_row - 1) != 0:
            row_factor = float(x_range) / (samples_per_row - 1)
        else:
            row_factor = 0
        if (samples_per_col - 1) != 0:
            col_factor = float(y_range) / (samples_per_col - 1)
        else:
            col_factor = 0
        col = idx / samples_per_col
        row = idx % samples_per_row

        x = int(start_x + row * row_factor)
        y = int(start_y + col * col_factor)

        return x, y

    def compute_orientations_for_point(self, image, x_center, y_center, window_width, window_height):
        orientations = []

        half_window_width = window_width / 2
        half_window_height = window_height / 2

        for x in range(self.cell_xres):
            orientations.append([])

            for y in range(self.cell_yres):
                orientations[x].append([])

                for o in range(self.cell_orient):
                    orientations[x][y].append(0)

        # alternative initialization to the above (but slower)
        # size = self.cell_xres, self.cell_yres, self.cell_orient
        # orientations = np.zeros(size, np.float)

        start_x = x_center - half_window_width
        end_x = x_center + half_window_width
        start_y = y_center - half_window_height
        end_y = y_center + half_window_height

        # import cv2
        # tmp = cv2.rectangle(image, (start_x, start_y), (end_x, end_y), 0, 1)
        # cv2.imshow('Image', image)
        # cv2.waitKey()

        if start_x < 1 or end_x > 255 or start_y < 1 or start_y > 255:
            raise "out of bounds"

        x_factor = 1.0 / (0.25 * float(end_x - start_x))
        y_factor = 1.0 / (0.25 * float(end_y - start_y))

        for x in range(start_x, end_x):
            x_index = int((x - start_x) * x_factor)

            if x < 0 or x >= image.shape[1] - 1:
                continue

            for y in range(start_y, end_y):
                y_index = int((y - start_y) * y_factor)

                if y < 0 or y >= image.shape[0] - 1:
                    continue

                # compute gradient
                # if black then assign 0 else assign 1
                #x_prev = image[y, x - 1] == 0 ?0:1;
                x_prev = 0 if image[y, x - 1] < 100 else 1
                # x_next = img.at < Vec3b > (y, x + 1)[0] == 0?0:1;
                x_next = 0 if image[y, x + 1] < 100 else 1
                # y_prev = img.at < Vec3b > (y - 1, x)[0] == 0?0:1;
                y_prev = 0 if image[y - 1, x] < 100 else 1
                # y_next = img.at < Vec3b > (y + 1, x)[0] == 0?0:1;
                y_next = 0 if image[y + 1, x] < 100 else 1

                x_diff = x_next - x_prev
                y_diff = y_next - y_prev

                if x_diff == -1:
                    if y_diff == -1:
                        orientations[x_index][y_index][0] += 1
                    elif y_diff == 0:
                        orientations[x_index][y_index][1] += 1
                    elif y_diff == 1:
                        orientations[x_index][y_index][2] += 1
                elif x_diff == 0:
                    if y_diff == -1:
                        orientations[x_index][y_index][3] += 1
                    elif y_diff == 1:
                        orientations[x_index][y_index][4] += 1
                elif x_diff == 1:
                    if y_diff == -1:
                        orientations[x_index][y_index][5] += 1
                    elif y_diff == 0:
                        orientations[x_index][y_index][6] += 1
                    elif y_diff == 1:
                        orientations[x_index][y_index][7] += 1

        return orientations

