import os

import cv2
import numpy as np


def load_image(img_in_dir, img_out_dir, img_path, label, *, gray_scale=False, invert=False):
    print("Now processing image:", img_path)
    file_name, file_extension = os.path.splitext(img_path)
    img_input = os.path.join(img_in_dir, img_path)
    img_output = os.path.join(img_out_dir, file_name + "-" + label + ".png")
    # LOAD IMAGE
    img = cv2.imread(img_input)
    if gray_scale or invert:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if invert:
        img = 255 - img
    return img, img_output


def save_image(img_output, image):
    cv2.imwrite(img_output, image.astype(np.uint8))
