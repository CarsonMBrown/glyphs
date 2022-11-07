import numpy as np
from numpy import mean


def mean_iou(bbox, bboxes):
    return mean([bbox.iou(other) for other in bboxes])


def get_unique_bboxes(bboxes, iou_threshold=.8):
    # only keep unique boxes
    unique_bboxes = []
    for i, a in enumerate(bboxes):
        duplicate = False
        for b in bboxes[i + 1:]:
            if a.iou(b) > iou_threshold:
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
        bbox for bbox in bboxes if
        dx_mean * x_min_percent <= (bbox.x_max - bbox.x_min) <= dx_mean * x_max_percent or
        dy_mean * y_min_percent <= (bbox.y_max - bbox.y_min) <= dy_mean * y_max_percent
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
    non_outliers = remove_bbox_outliers(bboxes, x_min_percent=x_min_percent, x_max_percent=x_max_percent,
                                        y_min_percent=y_min_percent, y_max_percent=y_max_percent)
    return [bbox for bbox in bboxes if bbox not in non_outliers]


def get_mean_dimensions(bboxes):
    dx_mean, dy_mean = mean(
        np.array([[bbox.x_max - bbox.x_min, bbox.y_max - bbox.y_min] for bbox in bboxes]).transpose(), axis=1)
    return dx_mean, dy_mean
