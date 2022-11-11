import os


# TODO coco output format
def write_coco(file_path, image_bboxes, category_map):
    """
    TODO not finished.

    Writes the passed in bounding boxes in coco format
    :param file_path: path to write file to
    :param image_bboxes: a dict of images_paths and the bounding boxes in that image
    :param category_map: dict to convert bbox classes (integers) to another representation
    :return: None
    """
    os.makedirs(file_path, exist_ok=True)
    cocojson = {"annotations": [], "images": [], "images": [], "categories": []}
    with open(file_path, encoding="UTF_8") as f:
        for image_path, bboxes in image_bboxes.items():
            for bbox in bboxes:
                annotation = {
                    "area": bbox.area,
                    "bbox": list(bbox.coco),
                    "category_id": bbox.get_class_index() if category_map is None else category_map[
                        bbox.get_class_index()],
                }


# TODO pascal output format
def write_pascal(file_path, image_bboxes):
    pass


# TODO csv format
def write_csv(file_path, image_bboxes):
    pass
