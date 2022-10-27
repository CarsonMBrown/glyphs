import os

import cv2
import numpy as np
from numpy.random import randint

color_index = -1
color_pallet = [
    (0, 0, 255),
    (0, 127, 255),
    (0, 255, 255),
    (0, 255, 0),
    (255, 0, 0),
    (130, 0, 75),
    (211, 0, 148)
]


def load_image(img_in_dir, img_out_dir, img_path, *, skip_existing=True, gray_scale=False, invert=False):
    print("Now processing image:", img_path)
    file_name, file_extension = os.path.splitext(img_path)
    img_input = os.path.join(img_in_dir, img_path)
    img_output = os.path.join(img_out_dir, file_name + ".png")
    # LOAD IMAGE
    if os.path.exists(img_output) and skip_existing:
        return None, img_output
    img = cv2.imread(img_input)
    if gray_scale or invert:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if invert:
        img = 255 - img
    return img, img_output


def save_image(img_output, image):
    cv2.imwrite(img_output, image.astype(np.uint8))


def random_color(in_order=True):
    if in_order:
        global color_index
        color_index += 1
        color_index %= len(color_pallet)
        return color_pallet[color_index]
    else:
        return randint(0, 256), randint(0, 256), randint(0, 256)


def plot_lines(img, lines, *, wait=0):
    """
    Given an image and a list of lists of points,
    plot each list of points as a line with a random color
    :param img:
    :param lines:
    :param wait:
    :return:
    """
    global color_index
    color_index = -1

    # Sort lines by y component of first point in first line
    lines.sort(key=lambda x: x[0][1])

    for line in lines:
        color = random_color()
        if len(line) > 1:
            for i, p in enumerate(line[:-1], 1):
                img = cv2.line(img, p, line[i], color=color, thickness=5)
        else:
            img = cv2.circle(img, line[0], 1, color=color, thickness=5)
    cv2.imshow("", img)
    cv2.waitKey(wait)


def plot_bboxes(img, bboxes, *, wait=0, color=None):
    """
    Given an image and a list of bounding boxes in pascal form (x_min, y_min, x_max, y_max),
    plot each bounding box as a rect with a random color.
    Edits image in place.
    :param img:
    :param bboxes:
    :param color:
    :param wait:
    :return:
    """
    for x_min, y_min, x_max, y_max in bboxes:
        img = cv2.rectangle(img, (x_min, y_min), (x_max, y_max),
                            color=random_color() if color is None else color,
                            thickness=2)
    if wait is None or wait >= 0:
        cv2.imshow("", img)
        cv2.waitKey(wait)
    return img
