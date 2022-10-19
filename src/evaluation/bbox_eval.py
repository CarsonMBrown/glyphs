from src.util.bbox_util import pascal_intersection_area, pascal_area


def intersection_over_union(truth, prediction):
    """
    AKA Pascal VOC method, taking bounding boxes in coco format.
    Returns the cardinality of the intersection of the bounding boxes
    over the cardinality of the union of the bounding boxes
    :param truth: bbox in COCO format (x_min, y_min, width, height)
    :param prediction: bbox in COCO format (x_min, y_min, width, height)
    :return: |truth region ∩ predicted region| / |truth region ∪ predicted region|
    """
    intersection_area = pascal_intersection_area(truth, prediction)
    IoU = intersection_area / (pascal_area(truth) + pascal_area(prediction) - intersection_area)
    return IoU
