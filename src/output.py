import csv
import os


# TODO coco output format
def write_coco(output_path, image_bboxes, category_map):
    """
    TODO not finished.

    Writes the passed in bounding boxes in coco format
    :param file_path: path to write file to
    :param image_bboxes: a dict of images_paths and the bounding boxes in that image
    :param category_map: dict to convert bbox classes (integers) to another representation
    :return: None
    """
    os.makedirs(output_path, exist_ok=True)
    cocojson = {"annotations": [], "images": [], "categories": []}
    with open(output_path, encoding="UTF_8") as f:
        for image_path, bboxes in image_bboxes.items():
            for bbox in bboxes:
                annotation = {
                    "area": bbox.area,
                    "bbox": list(bbox.coco),
                    "category_id": bbox.get_class_index() if category_map is None else category_map[
                        bbox.get_class_index()],
                }


# TODO pascal output format
def write_pascal(output_path, image_bboxes):
    pass


def write_csv(output_path, image_lines):
    with open(output_path, encoding="UTF_8", mode="w", newline="") as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow(["glyph", "certainty", "min_x", "min_y", "max_x", "max_y", "image_path", "line", "number"])
        for image, lines in image_lines:
            for i, line in enumerate(lines):
                for j, bbox in enumerate(line):
                    csvwriter.writerow(
                        [bbox.get_class(), bbox.get_class_certainty(), bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max,
                         image, i, j])
