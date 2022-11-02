import os

import cv2
import numpy as np
import pytesseract
from PIL import Image

from src.util.dir_util import get_input_img_paths, set_output_dir
from src.util.img_util import load_image


def tesseract(img_in_dir, img_out_dir):
    img_list = get_input_img_paths(img_in_dir)
    set_output_dir(img_out_dir)

    for img_path in img_list:
        img, img_output = load_image(img_in_dir, img_out_dir, img_path)

        kernel = np.ones((10, 10), np.uint8)

        # Using cv2.erode() method
        img = cv2.erode(img, kernel)

        # write the grayscale image to disk as a temporary file, so we can apply OCR to it
        filename = "{}.png".format(os.getpid())
        cv2.imwrite(filename, img)

        # load the image as a PIL/Pillow image, apply OCR, and then delete
        # the temporary file
        text = pytesseract.image_to_string(Image.open(filename), lang="grc")
        # print(pytesseract.image_to_boxes(Image.open(filename), lang="grc", output_type=Output.DICT))
        # pytesseract.image_to_data(Image.open(filename), lang="grc", output_type=Output.DICT)
        os.remove(filename)
        print(text)
