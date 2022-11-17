import csv
import json
import math
import os.path
from collections import defaultdict

import cv2
import numpy as np

from src.bounding.bound import get_minimal_bounding_boxes_v2
from src.util.bbox_util import BBox
from src.util.dir_util import split_image_name_extension, get_file_name
from src.util.glyph_util import glyph_to_name, glyph_to_glyph
from src.util.img_util import plot_lines


# Attributes of annotations:
# * id
# * image_id
# * category_id
# * bbox
# ? tags
# ? seg_id
def ann_id(a):
    return a["id"]


def ann_img_id(a):
    return a["image_id"]


def ann_cat_id(a):
    return a["category_id"]


def ann_x(a):
    return a["bbox"][0]


def ann_y(a):
    return a["bbox"][1]


def ann_w(a):
    return a["bbox"][2]


def ann_h(a):
    return a["bbox"][3]


def ann_basetype(a):
    return a["tags"]["BaseType"][0]


def ann_footmarktype(a):
    if "FootMarkType" in a["tags"]:
        return a["tags"]["FootMarkType"][+0]
    else:
        return ""


# Attributes of categories:
# * id
# * name
def cat_id(c):
    return c["id"]


def cat_name(c):
    return c["name"]


# Attributes of images:
# * id
# * file_name
def img_id(i):
    return i["id"]


def img_file(i):
    return i["file_name"]


class CocoReader:
    def __init__(self, in_dir):
        with open(os.path.join(in_dir, "HomerCompTrainingReadCoco.json"), encoding="utf-8") as json_file:
            admin = json.load(json_file)
        self.annotations = admin["annotations"]  # character bounding boxes
        self.categories = admin["categories"]  # character types
        self.images = admin["images"]  # images

        self.id_to_ann = {}
        for a in self.annotations:
            self.id_to_ann[ann_id(a)] = a

        self.id_to_cat = {}
        for c in self.categories:
            self.id_to_cat[cat_id(c)] = c

        self.id_to_img = {}
        for i in self.images:
            self.id_to_img[img_id(i)] = i

        self.img_to_ann = defaultdict(list)
        for a in self.annotations:
            self.img_to_ann[ann_img_id(a)].append(a)

        self.name_to_ann = defaultdict(list)
        for a in self.annotations:
            ident = ann_cat_id(a)
            cat = self.id_to_cat[ident]
            name = cat_name(cat)
            self.name_to_ann[name].append(a)

    def img_ids(self):
        return sorted(self.id_to_img.keys())

    def ann_name(self, a):
        return cat_name(self.id_to_cat[ann_cat_id(a)])

    def ann_class(self, a):
        return self.ann_name(a), ann_basetype(a), ann_footmarktype(a)

    def names(self):
        return sorted([cat_name(c) for c in self.categories])

    def occurring_names(self):
        return sorted(set([self.ann_name(a) for a in self.annotations]))

    def occurring_classes(self):
        return sorted(set([self.ann_class(a) for a in self.annotations]),
                      key=lambda c: c[0] + " " + c[1] + " " + c[2])


# Layout
def vert_span(a):
    return ann_y(a), ann_y(a) + ann_h(a)


def vert_spans_overlap(s1, s2):
    (l1, h1) = s1
    (l2, h2) = s2
    return h1 > l2 and h2 > l1


def overlapping_vert_span(spans, s):
    for i in range(len(spans)):
        if vert_spans_overlap(spans[i], s):
            return i
    return -1


def unify_vert_spans(s1, s2):
    (l1, h1) = s1
    (l2, h2) = s2
    return min([l1, l2]), max([h1, h2])


def divide_into_lines(ann):
    spans = []
    lines = defaultdict(list)
    ann = sorted(ann, key=lambda x: ann_x(x))
    for a in ann:
        span = vert_span(a)
        similar_index = overlapping_vert_span(spans, span)
        if similar_index >= 0:
            spans[similar_index] = unify_vert_spans(spans[similar_index], span)
        else:
            spans.append(span)
            similar_index = len(spans) - 1
        lines[similar_index].append(a)
    return [lines[i] for i in sorted(lines.keys(), key=lambda l: spans[l][0])]


def extract_glyphs(coco_dir, in_dir, out_dir, *, ocular_format=False, quality_filter=None,
                   glyphs_per_footmark_type_limit=None):
    if quality_filter is None:
        quality_filter = []
    coco = CocoReader(coco_dir)
    glyphs_per_footmark_type = {}

    metadata = [("image", "class", "origin_image", "base_type", "foot_mark_type")]

    for img, img_name, img_extension, _, annotations in get_image_and_data(coco, in_dir):
        for glyph_count, annotation in enumerate(annotations):
            glyph = annotation_to_glyph(annotation, coco)
            if glyph_to_name(glyph) == "period":
                continue
            tags = annotation["tags"]
            base_type = tags["BaseType"][0]
            if quality_filter and base_type not in quality_filter:
                continue
            foot_mark_type = None
            if "FootMarkType" in tags:
                foot_mark_type = tags["FootMarkType"][0]
                if len(tags["FootMarkType"]) > 1:
                    print(tags)
            if glyphs_per_footmark_type_limit is not None:
                if glyph not in glyphs_per_footmark_type:
                    glyphs_per_footmark_type[glyph] = {}
                if foot_mark_type not in glyphs_per_footmark_type[glyph]:
                    glyphs_per_footmark_type[glyph][foot_mark_type] = 1
                else:
                    glyphs_per_footmark_type[glyph][foot_mark_type] += 1
                if glyphs_per_footmark_type_limit < glyphs_per_footmark_type[glyph][foot_mark_type]:
                    continue

            x, y, dx, dy = annotation["bbox"]
            glyph_img = img[y:y + dy, x:x + dx]

            output_file_name = base_type + (
                "" if foot_mark_type is None else "-" + foot_mark_type) + "-" + img_name + str(glyph_count)

            if ocular_format:
                glyph_path = out_dir
            else:
                glyph_path = os.path.join(out_dir, glyph_to_name(glyph))
                if not os.path.exists(glyph_path):
                    os.mkdir(glyph_path)

            full_glyph_image_path = os.path.join(glyph_path, output_file_name)
            if ocular_format:
                with open(full_glyph_image_path + ".txt", mode="w", encoding="UTF_8") as meta_file:
                    meta_file.write(glyph)

            cv2.imwrite(full_glyph_image_path + img_extension, glyph_img)
            metadata.append((os.path.join(glyph_to_name(glyph), output_file_name + img_extension),
                             glyph,
                             img_name,
                             base_type,
                             foot_mark_type
                             ))
    with open(os.path.join(out_dir, "meta.csv"), mode="w", encoding="UTF_8", newline='') as meta_file:
        csv.writer(meta_file).writerows(metadata)


def extract_cropped_glyphs(coco_dir, in_dir, binarized_dir, out_dir, write_binary_guides=False):
    coco = CocoReader(coco_dir)
    for img, img_name, img_extension, _, annotations in get_image_and_data(coco, in_dir):
        binary_img = cv2.imread(os.path.join(binarized_dir, img_name + ".png"), cv2.IMREAD_GRAYSCALE)
        bboxes = []
        out_paths = []
        for glyph_count, annotation in enumerate(annotations):
            glyph = annotation_to_glyph(annotation, coco)
            if glyph_to_name(glyph) == "period":
                continue
            tags = annotation["tags"]
            base_type = tags["BaseType"][0]
            foot_mark_type = None
            if "FootMarkType" in tags:
                foot_mark_type = tags["FootMarkType"][0]

            bboxes.append(BBox.from_coco(*annotation["bbox"]))

            output_file_name = base_type + (
                "" if foot_mark_type is None else "-" + foot_mark_type) + "-" + img_name + str(glyph_count)

            glyph_path = os.path.join(out_dir, glyph_to_name(glyph))
            if not os.path.exists(glyph_path):
                os.makedirs(glyph_path)

            full_glyph_image_path = os.path.join(glyph_path, output_file_name)
            out_paths.append(full_glyph_image_path + ".png")

        for i, (bbox, path) in enumerate(zip(get_minimal_bounding_boxes_v2(binary_img, bboxes), out_paths)):
            cropped_img = bbox.crop(img)
            cv2.imwrite(path, cropped_img)
            if write_binary_guides:
                cropped_binary_img = cv2.cvtColor(bboxes[i].crop(binary_img), cv2.COLOR_GRAY2BGR)
                cropped_binary_img = cv2.rectangle(cropped_binary_img,
                                                   (bbox.x_min - bboxes[i].x_min, bbox.y_min - bboxes[i].y_min),
                                                   (bbox.x_max - bboxes[i].x_min, bbox.y_max - bboxes[i].y_min),
                                                   color=(0, 0, 255))
                cv2.imwrite(path.replace(".png", "_bin.png"), cropped_binary_img)


def generate_yolo_labels(coco_dir, out_dir, *, mono_class=False):
    coco = CocoReader(coco_dir)
    labels = set()
    # First pass, get all classes that actually appear
    for img_name, img_extension, _, annotations in get_image_data(coco):
        for annotation in annotations:
            glyph = glyph_to_name(annotation_to_glyph(annotation, coco))
            if not mono_class:
                labels.add(glyph)
    label_map = {}
    if not mono_class:
        labels = list(sorted(labels))
        for i, l in enumerate(labels):
            label_map[l] = i
            print(f"{i}: {l}")
    # Second pass, make all label files
    for img_name, img_extension, img_size, annotations in get_image_data(coco):
        with open(os.path.join(out_dir, img_name + ".txt"), mode="w") as f:
            for annotation in annotations:
                glyph = glyph_to_name(annotation_to_glyph(annotation, coco))
                bbox = BBox.from_coco(*annotation["bbox"])
                bbox_cx, bbox_cy, bbox_dx, bbox_dy = bbox.yolo(img_size)
                if not mono_class:
                    f.write(f"{label_map[glyph]} {bbox_cx} {bbox_cy} {bbox_dx} {bbox_dy} \n")
                else:
                    f.write(f"{0} {bbox_cx} {bbox_cy} {bbox_dx} {bbox_dy} \n")


def annotation_to_glyph(annotation, coco):
    return cat_name(coco.id_to_cat[annotation["category_id"]])


def get_image_data(coco):
    """
    :param coco:
    :return: (img_name, img_extension, annotations) for each image in the coco passed in
    """
    return [
        (*split_image_name_extension(image["img_url"]),
         (image["width"], image["height"]),
         get_annotations(coco, image))
        for image in coco.images]


def get_image_and_data(coco, in_dir):
    """
    :param coco:
    :param in_dir:
    :return: (img, img_name, img_extension, image_size, annotations) for each image in the coco passed in,
    with each img being taken from the in_dir
    """
    image_data = get_image_data(coco)
    imgs = []
    for img_name, img_extension, _, _ in image_data:
        if os.path.exists(os.path.join(in_dir, img_name + img_extension)):
            imgs.append(cv2.imread(os.path.join(in_dir, img_name + img_extension)))
        elif os.path.exists(os.path.join(in_dir, img_name + ".png")):
            imgs.append(cv2.imread(os.path.join(in_dir, img_name + ".png")))
        else:
            imgs.append(None)
    return [(imgs[i], *image_data[i]) for i in range(len(image_data)) if imgs[i] is not None]


def extract_text(coco_dir, in_dir, export_dir, show_images=False):
    coco = CocoReader(coco_dir)
    for image in coco.images:
        img_path = image["img_url"]
        img_name = get_file_name(img_path)

        centers, glyph_map = get_glyph_centers(coco, image)
        w_mean, _ = get_bbox_dim_means(coco, image)
        max_line_gap = 3 * w_mean
        _, h_sigma = get_bbox_dim_sigmas(coco, image)

        lines = []
        line = None
        for c in sorted(centers, key=lambda x: x[0]):
            new_line = True
            for line in lines:
                if c in line:
                    new_line = False
                    break
            closest_points = [p for p in get_points_by_distance(c, centers, h_sigma)]
            if new_line:
                line = [c]
            else:
                lines.remove(line)
            if new_line or line.index(c) == len(line) - 1:
                expected_dy = 0 if new_line else np.average(
                    [line[i][1] - line[i + 1][1] for i in range(len(line) - 1)])
                right_closest = [p for p in
                                 [p1 for p1 in closest_points if p1[1] - (expected_dy / h_sigma) < 2]
                                 if p[2][0] > c[0] and p[2][0] - c[0] < max_line_gap]
                if right_closest:
                    line.append(right_closest[0][2])
            lines.append(line)

        lines = merge_lines(lines, max_line_gap)
        if show_images:
            plot_lines(cv2.imread(os.path.join(in_dir, img_path)), lines, wait=0)
        with open(os.path.join(export_dir, img_name + ".txt"), encoding="UTF_8", mode="w") as f:
            f.write("".join(lines_to_glyphs(lines, glyph_map)))


def get_glyph_centers(coco, image):
    glyph_centers = [(
        get_bounding_box_center(*annotation["bbox"]),
        glyph_to_glyph(cat_name(coco.id_to_cat[annotation["category_id"]])))
        for annotation in get_annotations(coco, image)
    ]
    glyph_map = {}
    for gc in glyph_centers:
        glyph_map[gc[0]] = gc[1]
    return [gc[0] for gc in glyph_centers], glyph_map


def get_annotations(coco, image):
    return [a for a in coco.annotations if a["image_id"] == image["id"]]


def get_bounding_box_center(x, y, w, h):
    return x + w // 2, y + h // 2


def square_y_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def get_points_by_distance(p, ps, h_sigma):
    return sorted([(*get_bbox_dist_and_dy_sigma_from_point(p, pn, h_sigma), pn) for pn in ps if p != pn],
                  key=lambda z: z[0])


def merge_lines(lines, max_line_gap):
    # No do-while loops, so have to force it
    merging = True
    while merging:
        merging = False
        merged = []
        new_lines = []
        # Get centroids of each lines
        centroids = np.array([get_centroid(line) for line in lines])
        # get stddev of dy between adjacent lines
        line_dy_sigma = np.std(get_deltas(sorted(centroids.swapaxes(0, 1)[1])))
        line_centroid = list(zip(lines, centroids))

        for i, (line1, centroid1) in enumerate(line_centroid):
            if i in merged:
                continue
            new_line = line1
            for j, (line2, centroid2) in enumerate(line_centroid[i + 1:], i + 1):
                if j in merged:
                    continue
                if y_dist(centroid1, centroid2) / line_dy_sigma < .5 and \
                        (x_dist(line1[0], line2[-1]) < max_line_gap or x_dist(line1[-1], line2[0]) < max_line_gap):
                    new_line += line2
                    merged += [i, j]
                    merging = True
            new_lines.append(sorted(new_line, key=lambda x: x[0]))
        merged = []
        lines = []
        # merge lines that touch
        for i, line1 in enumerate(new_lines):
            if i in merged:
                continue
            new_line = line1
            for j, line2 in enumerate(new_lines[i + 1:], i + 1):
                if j in merged:
                    continue
                if not set(line1).isdisjoint(set(line2)):
                    new_line = sorted(list(set(new_line).union(set(line2))), key=lambda x: x[0])
                    merged += [i, j]
                    merging = True
            lines.append(new_line)
    return lines


def get_centroid(points):
    points = np.array(points)
    length = points.shape[0]
    sum_x = np.sum(points[:, 0])
    sum_y = np.sum(points[:, 1])
    return sum_x / length, sum_y / length


def get_deltas(vals):
    return [vals[i] - vals[i + 1] for i in range(len(vals) - 1)]


def x_dist(p1, p2):
    return abs(p1[0] - p2[0])


def y_dist(p1, p2):
    return abs(p1[1] - p2[1])


def lines_to_glyphs(lines, glyph_map):
    lines.sort(key=lambda x: get_centroid(x)[1])
    return [glyph_map[point] for line in lines for point in line]


def get_bbox_dims(coco, image):
    # get bounding boxs
    bboxes = np.array([annotation["bbox"] for annotation in get_annotations(coco, image)])
    # return lists of bbox width and height by rotating the list of bounding boxes
    return bboxes.swapaxes(0, 1)[2:]


def get_bbox_dist_and_angle_from_point(p1, p2):
    return square_y_distance(p1, p2), abs(
        math.degrees(math.atan((p1[1] - p2[1]) / (p1[0] - p2[0]))) if p1[0] != p2[0] else 0)


def get_bbox_dist_and_dy_sigma_from_point(p1, p2, h_sigma):
    return math.dist(p1, p2), abs(p1[1] - p2[1]) / h_sigma


def get_bbox_dim_means(coco, image):
    # get widths and heights of bboxes
    ws, hs = get_bbox_dims(coco, image)
    # return means
    return ws.mean(), hs.mean()


def get_bbox_dim_sigmas(coco, image):
    # return std-devs
    ws, hs = get_bbox_dims(coco, image)
    # get widths and heights of bboxes
    return ws.std(), hs.std()
