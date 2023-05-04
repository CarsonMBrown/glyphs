import math
from statistics import mean
from uuid import uuid4

import numpy as np
from scipy.special import softmax

from src.util.glyph_util import index_to_glyph, glyph_to_index, get_num_classes


class BBox:
    width, height, area, center = None, None, None, None
    uuid = None
    probabilities = None
    confidence = None

    def __init__(self, x_min, y_min, x_max, y_max, *, probabilities=None, uuid=None, confidence=None):
        """Creates a bounding box using the pascal format and an optional list of class probabilities and uuid"""
        self.x_min = int(x_min)
        self.y_min = int(y_min)
        self.x_max = int(x_max)
        self.y_max = int(y_max)
        # Precalculate some commonly used values
        self.recalculate_reference_values()
        # init probabilities and uuid
        self.probabilities = probabilities
        self.confidence = confidence
        if uuid is None:
            self.uuid = uuid4()

    def offset(self, x, y):
        self.x_min += x
        self.x_max += x
        self.y_min += y
        self.y_max += y
        self.center = self.calc_center()

    def grow_to(self, width, height):
        if self.width < width:
            self.set_width(width)
        if self.height < height:
            self.set_height(height)

    def calc_dims(self):
        return self.x_max - self.x_min, self.y_max - self.y_min

    def calc_area(self):
        return (self.width + 1) * (self.height + 1)

    def calc_center(self):
        return int((self.x_min + self.x_max) // 2), int((self.y_min + self.y_max) // 2)

    def recalculate_reference_values(self):
        self.width, self.height = self.calc_dims()
        self.area = self.calc_area()
        self.center = self.calc_center()

    def is_valid(self):
        return self.width > 0 and self.height > 0

    @staticmethod
    def from_coco(x_min, y_min, width, height):
        return BBox(x_min, y_min, x_min + width, y_min + height)

    def pascal(self):
        return self.x_min, self.y_min, self.x_max, self.y_max

    def coco(self):
        return self.x_min, self.y_min, self.width, self.height

    def yolo(self, img_size):
        img_x, img_y = img_size
        return (
            (self.x_min + self.width / 2) / img_x,
            (self.y_min + self.height / 2) / img_y,
            self.width / img_x,
            self.height / img_y
        )

    def set_width(self, new_width):
        cx, cy = self.center
        self.x_min = cx - new_width // 2
        self.x_max = self.x_min + new_width
        self.recalculate_reference_values()

    def set_height(self, new_height):
        cx, cy = self.center
        self.y_min = cy - new_height // 2
        self.y_max = self.y_min + new_height
        self.recalculate_reference_values()

    def trim(self, amount, *, side="r"):
        """
        Trims some amount of pixels off the given side
        :param amount: how much to trim off of given side
        :param side: right, left, top, bottom
        :return: None
        """
        if side == "r":
            self.x_max -= amount
        elif side == "l":
            self.x_min += amount
        elif side == "b":
            self.y_max -= amount
        elif side == "t":
            self.y_min += amount

        self.recalculate_reference_values()

    def copy(self):
        return BBox(self.x_min, self.y_min, self.x_max, self.y_max, probabilities=self.probabilities,
                    uuid=self.uuid, confidence=self.confidence)

    def is_copy(self, other: "BBox"):
        if not isinstance(other, BBox):
            raise TypeError
        return self.uuid == other.uuid

    def add_class_probabilities(self, probabilities):
        self.probabilities = probabilities

    def get_class_probabilities(self):
        return self.probabilities

    def get_class_index(self):
        if self.probabilities is None:
            return None
        return np.argmax(self.probabilities)

    def get_class(self):
        return index_to_glyph(self.get_class_index())

    def get_class_certainty(self):
        return softmax(self.probabilities)[self.get_class_index()]

    def set_class(self, glyph, n_classes=0):
        if n_classes == 0:
            n_classes = get_num_classes()
        self.probabilities = [0] * n_classes
        self.probabilities[glyph_to_index(glyph)] = 1
        return self

    def intersections(self, other: "BBox"):
        """Return the intersection of the x and y dimensions between two bboxes"""
        if not isinstance(other, BBox):
            raise TypeError

        inter_min_x, inter_max_x = max(self.x_min, other.x_min), min(self.x_max, other.x_max)
        inter_min_y, inter_max_y = max(self.y_min, other.y_min), min(self.y_max, other.y_max)
        return max(0, inter_max_x - inter_min_x + 1), max(0, inter_max_y - inter_min_y + 1)

    def is_inside(self, other: "BBox"):
        """Return True if this bbox is contained in the other bbox, False otherwise,
        allowing the case in which the bounding boxes have the same edge(s)."""
        if not isinstance(other, BBox):
            raise TypeError
        return (self.x_min >= other.x_min and self.x_max <= other.x_max and
                self.y_min >= other.y_min and self.y_max <= other.y_max)

    def is_intersecting(self, other: "BBox", *, x, y, area):
        """Return True if this bbox's x y and area intersection (as a percentage) are all above the given values,
        false otherwise."""
        if not isinstance(other, BBox):
            raise TypeError
        x_inter, y_inter = self.intersections(other)
        a_inter = x_inter * y_inter / self.area
        return a_inter >= area and x_inter / self.width > x and y_inter / self.height > y

    def get_enclosing(self, others):
        return [other for other in others if self != other and self.is_inside(other)]

    def get_overlaid(self, others, x, y, area):
        return [other for other in others if self != other and
                self.is_intersecting(other, x=x, y=y, area=area) and
                other.contains_center(self)]

    def contains_point(self, point, *, dimension_wise=False, allow_border=True):
        """
        Checks if the given point is inside the bounding box,
        by default allowing for points on the border to be considered inside
        :param point: (x,y)
        :param dimension_wise: if true will return a pair of bools representing if the point is contained in the x dim
         of the bounding box, and if the point is contained in the y dim of the bounding box
        :param allow_border: if a point on the border is contained
        :return: a bool representing if the point is contained in the bounding box
        """
        x, y = point
        in_x = (self.x_min <= x <= self.x_max) if allow_border else (self.x_min < x < self.x_max)
        in_y = (self.y_min <= y <= self.y_max) if allow_border else (self.y_min < y < self.y_max)
        if dimension_wise:
            return in_x, in_y
        return in_x and in_y

    def contains_center(self, other: "BBox", *, dimension_wise=False, allow_border=True):
        if not isinstance(other, BBox):
            raise TypeError
        return self.contains_point(other.center, dimension_wise=dimension_wise, allow_border=allow_border)

    def iou(self, other: "BBox"):
        """
        AKA Pascal VOC method.
        Returns the cardinality of the intersection of the bounding boxes
        over the cardinality of the union of the bounding boxes
        :return: |truth region ∩ predicted region| / |truth region ∪ predicted region|
        """
        if not isinstance(other, BBox):
            raise TypeError
        inter_x, inter_y = self.intersections(other)
        intersection_area = inter_x * inter_y
        IoU = intersection_area / (self.area + other.area - intersection_area)
        return IoU

    def dimensional_iou(self, other: "BBox"):
        """
        A modified version of intersection over union, AKA Pascal VOC method.
        Returns the cardinality of the intersection of the bounding boxes
        over the cardinality of the union of the bounding boxes for both the x and y dimensions
        :return: the iou for the x and y dimensions as a tuple (x,y)
        """
        inter_x, inter_y = self.intersections(other)
        union_x = (linear_combined_area(self.x_max, other.x_max, self.x_min, other.x_min) - inter_x)
        union_y = (linear_combined_area(self.y_max, other.y_max, self.y_min, other.y_min) - inter_y)
        return (inter_x / union_x), (inter_y / union_y)

    def get_pair(self, others):
        """Returns a tuple containing the bbox and iou of that bbox = (bbox, iou)"""
        bbox_iou_pairs = [(other, self.iou(other)) for other in others]
        bbox_iou_pairs.sort(key=lambda x: x[1], reverse=True)
        if len(bbox_iou_pairs) > 0:
            return bbox_iou_pairs[0]
        return None, 0

    def get_intersection_angle(self, other: "BBox"):
        """Return the angle (in degrees) between the two bounding box centers"""
        if not isinstance(other, BBox):
            raise TypeError

        (x1, y1), (x2, y2) = self.center, other.center
        dot_product = x1 * x2 + y1 * y2
        cosine_angle = dot_product / (math.sqrt(x1 * x1 + y1 * y1) * math.sqrt(x2 * x2 + y2 * y2))
        return math.degrees(math.acos(cosine_angle))

    def edge_distance(self, other):
        """Returns the x and y distances (dx,dy) between the edges of two bboxes.
        Only defined for non-intersecting boxes."""
        x_dist = max(other.x_min - self.x_max, self.x_min - other.x_max)
        y_dist = max(other.y_min - self.y_max, self.y_min - other.y_max)
        return x_dist, y_dist

    def relative_edge_distance(self, other):
        """Returns the x and y distances (dx,dy) between the edges of two bboxes as the
        percentage of the mean width/height of the two bounding boxes.
        Only defined for non-intersecting boxes."""
        mean_width = (self.width + other.width) / 2
        mean_height = (self.height + other.height) / 2
        x_dist, y_dist = self.edge_distance(other)
        return x_dist / mean_width, y_dist / mean_height

    def crop(self, img):
        x_min, y_min = max(0, self.x_min), max(0, self.y_min)
        x_max, y_max = min(img.shape[1], self.x_max + 1), min(img.shape[0], self.y_max + 1)
        return img[y_min:y_max, x_min:x_max]

    def merge(self, other):
        return BBox(
            min(self.x_min, other.x_min),
            min(self.y_min, other.y_min),
            max(self.x_max, other.x_max),
            max(self.y_max, other.y_max)
        )

    def split(self):
        return (
            BBox(self.x_min, self.y_min, self.x_min + self.width // 2, self.y_max),
            BBox(self.x_min + self.width // 2 + 1, self.y_min, self.x_max, self.y_max)
        )

    def __lt__(self, other):
        return self.x_min < other.x_min or (self.x_min == other.x_min and self.y_min > other.y_min)

    def __gt__(self, other):
        return self.x_min > other.x_min or (self.x_min == other.x_min and self.y_min < other.y_min)

    def __le__(self, other):
        return self.x_min >= other.x_min or (self.x_min == other.x_min and self.y_min <= other.y_min)

    def __ge__(self, other):
        return self.x_min >= other.x_min or (self.x_min == other.x_min and self.y_min <= other.y_min)

    def __eq__(self, other):
        return self.x_min == other.x_min and self.center == other.center and self.area == self.area

    def __repr__(self):
        if self.probabilities is None:
            return str(self.pascal())
        return str(self.pascal()) + ":" + self.get_class()

    def __str__(self):
        if self.probabilities is None:
            return str(self.pascal())
        return str(self.pascal()) + ":" + self.get_class()


def bboxes_to_crops(bboxes, img):
    """
    Converts a list of bounding boxes to a list of images
    :param bboxes: the bounding boxes to convert to cropped images
    :param img: img to crop from
    :return: list of images
    """
    return [bbox.crop(img) for bbox in bboxes]


def linear_combined_area(max1, max2, min1, min2):
    """
    Returns union + intersection between two lines
    :param max1:
    :param max2:
    :param min1:
    :param min2:
    :return:
    """
    return max1 + max2 - min1 - min2


def get_mean_dims(bboxes):
    """Return mean (as int) bbox width and height for a line"""
    if len(bboxes) > 0:
        mean_width = int(mean([bbox.width for bbox in bboxes]))
        mean_height = int(mean([bbox.height for bbox in bboxes]))
        return mean_width, mean_height
    return 0, 0


def get_bbox_pairs(bboxes):
    return zip(bboxes[:-1], bboxes[1:])
