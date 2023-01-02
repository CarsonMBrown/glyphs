import csv
import json


def write_coco(output_name, image_lines):
    """
    TODO: finish this
    Writes the given image, line pairs to csv
    :param output_name: name of file to output to
    :param image_lines: a list of image,lines pairs
    :return: None
    """
    for image, lines in image_lines:
        for i, line in enumerate(lines):
            for j, bbox in enumerate(line):
                csvwriter.writerow(
                    [bbox.get_class(), bbox.get_class_certainty(), bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max,
                     image, i, j])

    with open(output_name + ".json", encoding="UTF_8", mode="w", newline="") as f:
        f.write(json.dumps(coco))


def write_csv(output_name, image_lines):
    """
    Writes the given image, line pairs to csv
    :param output_name: name of file to output to
    :param image_lines: a list of image,lines pairs
    :return: None
    """
    with open(output_name + ".csv", encoding="UTF_8", mode="w", newline="") as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow(["glyph", "certainty", "min_x", "min_y", "max_x", "max_y", "image_path", "line", "number"])
        for image, lines in image_lines:
            for i, line in enumerate(lines):
                for j, bbox in enumerate(line):
                    csvwriter.writerow(
                        [bbox.get_class(), bbox.get_class_certainty(), bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max,
                         image, i, j])
