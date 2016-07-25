
from skimage.feature import hog
import math
import numpy as np
from feature_extractor import BaseFeatureExtractor


class SKImageHOGFeatureExtractor(BaseFeatureExtractor):

    def __init__(self, params):
        super(SKImageHOGFeatureExtractor, self).__init__(params)

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
            self.window_ratio = 0.125

    def _compute(self, image):
        cols = image.shape[1]
        rows = image.shape[0]

        window_width = int(self.window_ratio * float(cols))
        window_height = int(self.window_ratio * float(rows))

        size = self.feature_count, self.cell_xres * self.cell_yres * self.cell_orient
        descriptor = np.zeros(size, dtype=np.float)

        for idx in range(self.feature_count):
            x, y = self.get_keypoint_centre(cols, rows, window_width, window_height, idx)

            s_col = x - window_width / 2
            e_col = s_col + window_width

            s_row = y - window_height / 2
            e_row = s_row + window_height

            sub_image = image[s_row:e_row, s_col:e_col]

            fv = self.extract_features(sub_image,
                                       orientations=self.cell_orient,
                                       pixels_per_cell=window_width/self.cell_xres)

            descriptor[idx] = fv

        return descriptor

    def extract_features(self, img, orientations=8, pixels_per_cell=8, cells_per_block=1):
        fd = hog(img,
                 orientations=orientations,
                 pixels_per_cell=(pixels_per_cell, pixels_per_cell),
                 cells_per_block=(cells_per_block, cells_per_block),
                 feature_vector=True,
                 visualise=False)

        return fd

    def get_keypoint_centre(self, cols, rows, window_width, window_height, idx):

        half_window_width = window_width / 2
        half_window_height = window_height / 2

        start_x = half_window_width
        end_x = cols - half_window_width
        start_y = half_window_height
        end_y = rows - half_window_height

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
