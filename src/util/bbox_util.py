import math


def pascal_area(bbox):
    """
    get area of pascal bbox area (x_min, y_min, x_max, y_max)
    :param bbox: bounding box in pascal form (x_min, y_min, x_max, y_max)
    :return:
    """
    x_min, y_min, x_max, y_max = bbox
    return (x_max - x_min + 1) * (y_max - y_min + 1)


def pascal_intersections(bbox1, bbox2):
    """
    get intersection of the x and y dimensions between two pascal bboxes (x_min, y_min, x_max, y_max)
    :param bbox1: bbox in pascal form
    :param bbox2: bbox in pascal form
    :return:
    """
    inter_min_x, inter_max_x = max(bbox1[0], bbox2[0]), min(bbox1[2], bbox2[2])
    inter_min_y, inter_max_y = max(bbox1[1], bbox2[1]), min(bbox1[3], bbox2[3])
    return max(0, inter_max_x - inter_min_x + 1), max(0, inter_max_y - inter_min_y + 1)


def bbox_inside(bbox1, bbox2):
    """
    checks if for two pascal bboxes (x_min, y_min, x_max, y_max), the first in inside the second
    :param bbox1:
    :param bbox2:
    :return: True if bbox1 is fully contained by bbox2, False otherwise,
    allowing the case in which the bounding boxes have the same edge(s)
    """
    x_min1, y_min1, x_max1, y_max1 = bbox1
    x_min2, y_min2, x_max2, y_max2 = bbox2
    return x_min1 >= x_min2 and y_min1 >= y_min2 and x_max1 <= x_max2 and y_max1 <= y_max2


def pascal_to_coco(bbox):
    """
    Converts bounding box in pascal format to coco format.
    :param bbox: bounding box in pascal format (x_min, y_min, x_max, y_max)
    :return: bounding box in coco format (x_min, y_min, width, height)
    """
    x_min, y_min, x_max, y_max = bbox
    return x_min, y_min, x_max - x_min, y_max - y_min


def coco_to_pascal(bbox):
    """
    Converts bounding box in coco format to pascal format.
    :param bbox: bounding box in coco format (x_min, y_min, width, height)
    :return: bounding box in pascal format (x_min, y_min, x_max, y_max)
    """
    x_min, y_min, width, height = bbox
    return x_min, y_min, x_min + width, y_min + height


def coco_to_yolo(bbox, img_size):
    """
    Converts bounding box in coco format to yolo format.
    :param bbox: bounding box in coco format (x_min, y_min, width, height)
    :param img_size: size of image (x, y) in pixels
    :return: bounding box in pascal format (x_min, y_min, x_max, y_max)
    """
    x_min, y_min, width, height = bbox
    img_x, img_y = img_size
    return (x_min + width / 2) / img_x, (y_min + height / 2) / img_y, width / img_x, height / img_y


def bboxes_to_images(bboxes, img):
    """
    Converts a list of pascal formatted bounding boxes to a list of images
    :param bboxes: pascal formatted bounding boxes to convert to images
    :param img: img to crop from
    :return: list of images
    """
    return [img[min_y:max_y, min_x:max_x] for min_x, min_y, max_x, max_y in bboxes]


def bbox_center(bbox):
    """
    :return: the center of a pascal formatted (x_min, y_min, x_max, y_max) bounding box
    """
    x_min, y_min, x_max, y_max = bbox
    return (x_max - x_min), (y_max - y_min)


def bbox_intersection_angle(bbox1, bbox2):
    """
    Returns the angle between bounding boxes
    :param bbox1:
    :param bbox2:
    :return:
    """
    (x1, y1), (x2, y2) = bbox_center(bbox1), bbox_center(bbox2)
    dot_product = x1 * x2 + y1 * y2
    cosine_angle = dot_product / (math.sqrt(x1 * x1 + y1 * y1) * math.sqrt(x2 * x2 + y2 * y2))
    return math.degrees(math.acos(cosine_angle))


def iou(bbox1, bbox2):
    """
    AKA Pascal VOC method, taking bounding boxes in coco format.
    Returns the cardinality of the intersection of the bounding boxes
    over the cardinality of the union of the bounding boxes
    :param bbox1: bbox in pascal format  (x_min, y_min, x_max, y_max)
    :param bbox2: bbox in pascal format  (x_min, y_min, x_max, y_max)
    :return: |truth region ∩ predicted region| / |truth region ∪ predicted region|
    """
    inter_x, inter_y = pascal_intersections(bbox1, bbox2)
    intersection_area = inter_x * inter_y
    IoU = intersection_area / (pascal_area(bbox1) + pascal_area(bbox2) - intersection_area)
    return IoU


def dimensional_iou(bbox1, bbox2):
    """
    A modified version of intersection over union, AKA Pascal VOC method, taking bounding boxes in pascal format.
    Returns the cardinality of the intersection of the bounding boxes
    over the cardinality of the union of the bounding boxes for both the x and y dimensions
    :param bbox1:  bbox in pascal format  (x_min, y_min, x_max, y_max)
    :param bbox2: bbox in pascal format  (x_min, y_min, x_max, y_max)
    :return:
    """
    x_min1, y_min1, x_max1, y_max1 = bbox1
    x_min2, y_min2, x_max2, y_max2 = bbox2
    inter_x, inter_y = pascal_intersections(bbox1, bbox2)
    union_x = (linear_intersection(x_max1, x_max2, x_min1, x_min2) - inter_x)
    union_y = (linear_intersection(y_max1, y_max2, y_min1, y_min2) - inter_y)
    return (inter_x / union_x), (inter_y / union_y)


def point_in_bbox(point, bbox, *, dimension_wise=False, allow_border=True):
    """
    :param point: (x,y)
    :param bbox: bbox in pascal format  (x_min, y_min, x_max, y_max)
    :param dimension_wise: if true will return a pair of bools representing if the point is contained in the x dim
     of the bounding box, and if the point is contained in the y dim of the bounding box
    :param allow_border: if a point on the border is contained
    :return: a bool representing if the point is contained in the bounding box
    """
    x, y = point
    x_min, y_min, x_max, y_max = bbox
    in_x = (x_min <= x <= x_max) if allow_border else (x_min < x < x_max)
    in_y = (y_min <= y <= y_max) if allow_border else (y_min < y < y_max)
    if dimension_wise:
        return in_x, in_y
    return in_x and in_y


def linear_intersection(max1, max2, min1, min2):
    return max1 + max2 - min1 - min2
