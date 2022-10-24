def pascal_area(bbox):
    """
    get area of pascal bbox area
    :param bbox:
    :return:
    """
    x_min, y_min, x_max, y_max = bbox
    return (x_max - x_min + 1) * (y_max - y_min + 1)


def pascal_intersection_area(bbox1, bbox2):
    """
    get area of the intersection between two pascal bboxes
    :param bbox1:
    :param bbox2:
    :return:
    """
    inter_min_x, inter_max_x = max(bbox1[0], bbox2[0]), min(bbox1[2], bbox2[2])
    inter_min_y, inter_max_y = max(bbox1[1], bbox2[1]), min(bbox1[3], bbox2[3])
    inter_w, inter_y = max(0, inter_max_x - inter_min_x + 1), max(0, inter_max_y - inter_min_y + 1)
    return inter_w * inter_y


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
