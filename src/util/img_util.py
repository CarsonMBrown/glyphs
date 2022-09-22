import os

import cv2
import numpy as np


def load_image(img_in_dir, img_out_dir, img_path, label):
    print("Now processing image:", img_path)
    file_name, file_extension = os.path.splitext(img_path)
    img_input = os.path.join(img_in_dir, img_path)
    img_output = os.path.join(img_out_dir, file_name + "-" + label + ".png")
    # LOAD IMAGE
    img = cv2.imread(img_input)
    return img, img_output


def save_image(img_output, image):
    cv2.imwrite(img_output, image.astype(np.uint8))
