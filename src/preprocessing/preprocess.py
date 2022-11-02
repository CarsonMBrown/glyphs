from math import floor

import cv2

from src.util.dir_util import get_input_img_paths, set_output_dir
from src.util.img_util import load_image, save_image


def scale(img_in_dir, img_out_dir, max_w, max_h, binarize=False):
    img_list = get_input_img_paths(img_in_dir)
    set_output_dir(img_out_dir)

    for img_path in img_list:
        img, img_output = load_image(img_in_dir, img_out_dir, img_path, skip_existing=False)
        if img is None:
            continue
        h, w, _ = img.shape
        # if image size less than max size, do nothing
        if w > max_w or h > max_h:
            if w / max_w >= h / max_h:
                img = cv2.resize(img, (max_w, floor(h / w * max_h)))
            else:
                img = cv2.resize(img, (floor(w / h * max_w), max_h))
        save_image(img_output, img)
