import math

import numpy as np

from src.util.glyph_util import index_to_glyph


class BBox:
    def __init__(self, x_min, y_min, x_max, y_max):
        """Creates a bounding box using the pascal format."""
        self.x_min = int(x_min)
        self.y_min = int(y_min)
        self.x_max = int(x_max)
        self.y_max = int(y_max)
        self.width = x_max - x_min
        self.height = y_max - y_min
        self.area = (self.width + 1) * (self.height + 1)
        self.center = int((x_min + x_max) // 2), int((y_min + y_max) // 2)
        self.probabilities = None

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
        return self.x_min >= other.x_min and self.y_min >= other.y_min and self.x_max <= other.x_max and self.y_max <= other.y_max

    def contains_point(self, point, *, dimension_wise=False, allow_border=True):
        """
        Checks if the given point is inside the bounding box,
        by default allowing for points on the border to be considered inside.
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

    def get_intersection_angle(self, other: "BBox"):
        """Return the angle (in degrees) between the two bounding box centers"""
        if not isinstance(other, BBox):
            raise TypeError

        (x1, y1), (x2, y2) = self.center, other.center
        dot_product = x1 * x2 + y1 * y2
        cosine_angle = dot_product / (math.sqrt(x1 * x1 + y1 * y1) * math.sqrt(x2 * x2 + y2 * y2))
        return math.degrees(math.acos(cosine_angle))

    def relative_edge_distance(self, other):
        """Returns the x and y distances (dx,dy) between the edges of two bboxes as the
        percentage of the mean width/height of the two bounding boxes.
        Only defined for non-intersecting boxes."""
        mean_width = (self.x_max + other.x_max - self.x_min - other.x_min)
        mean_height = (self.y_max + other.y_max - self.y_min - other.y_min)
        x_dist = max(other.x_min - self.x_max, self.x_min - other.x_max) / mean_width
        y_dist = max(other.y_min - self.y_max, self.y_min - other.y_max) / mean_height
        return x_dist, y_dist

    def crop(self, img):
        return img[self.y_min:self.y_max + 1, self.x_min:self.x_max + 1]

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
        return str(self.pascal(), self.get_class_probabilities())


def bboxes_to_crops(bboxes, img):
    """
    Converts a list of bounding boxes to a list of images
    :param bboxes: bounding boxes to convert to images
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
