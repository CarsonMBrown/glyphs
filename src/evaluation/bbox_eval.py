import numpy as np
from numpy import mean

from src.util.bbox_util import iou


def mean_iou(bbox, bboxes):
    return mean([iou(bbox, bbox2) for bbox2 in bboxes])


def get_unique_bboxes(bboxes, iou_threshold=.8):
    # only keep unique boxes
    unique_bboxes = []
    for i, a in enumerate(bboxes):
        duplicate = False
        for b in bboxes[i + 1:]:
            if iou(a, b) > iou_threshold:
                duplicate = True
        if not duplicate:
            unique_bboxes.append(a)
    return unique_bboxes


def remove_bbox_outliers(bboxes, *, x_min_percent=.5, x_max_percent=2, y_min_percent=.5, y_max_percent=2):
    """
    Removes all bounding boxes where the width or height of the bounding box is more/less than the allowed
    max/min percentage of the mean width/height.
    :param bboxes:
    :param x_min_percent:
    :param x_max_percent:
    :param y_min_percent:
    :param y_max_percent:
    :return:
    """
    dx_mean, dy_mean = get_mean_dimensions(bboxes)
    return [
        (x_min, y_min, x_max, y_max) for x_min, y_min, x_max, y_max in bboxes if
        dx_mean * x_min_percent <= (x_max - x_min) <= dx_mean * x_max_percent or
        dy_mean * y_min_percent <= (y_max - y_min) <= dy_mean * y_max_percent
    ]


def get_bbox_outliers(bboxes, *, x_min_percent=.5, x_max_percent=2, y_min_percent=.5, y_max_percent=2):
    """
    Gets all bounding boxes where the width or height of the bounding box is more/less than the allowed
    max/min percentage of the mean width/height.
    :param bboxes:
    :param x_min_percent:
    :param x_max_percent:
    :param y_min_percent:
    :param y_max_percent:
    :return:
    """
    dx_mean, dy_mean = get_mean_dimensions(bboxes)
    return [
        (x_min, y_min, x_max, y_max) for x_min, y_min, x_max, y_max in bboxes if
        not (dx_mean * x_min_percent <= (x_max - x_min) <= dx_mean * x_max_percent) and
        not (dy_mean * y_min_percent <= (y_max - y_min) <= dy_mean * y_max_percent)
    ]


def get_mean_dimensions(bboxes):
    dx_mean, dy_mean = mean(
        np.array([[x_max - x_min, y_max - y_min] for x_min, y_min, x_max, y_max in bboxes]).transpose(),
        axis=1)
    return dx_mean, dy_mean
