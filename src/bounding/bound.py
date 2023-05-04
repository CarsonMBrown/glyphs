from src.bounding.connected_components import get_connected_component_bounding_boxes, bound_and_render
from src.util.bbox_util import BBox, get_mean_dims


def export_connected_component(img_in_dir, img_out_dir):
    """
    Bound using connected components and then save images
    :param img_in_dir: glyph_path to directory containing input images
    :param img_out_dir: glyph_path to directory to output images
    :return: None
    """
    print(bound_and_render(img_in_dir, img_out_dir))


def get_minimal_bounding_boxes(binary_img, bboxes, *, split=False):
    """
    Crops the given bounding boxes to contain only the ink, given the binary image.
    Bounding boxes with no ink are left as they are and returned separately
    :param binary_img: black and white image where the black represents ink
    :param bboxes: the bounding boxes to crop to the minimal size while retaining all
    the ink that was fully in the bounding box already
    :param split: if the cropped and non-cropped should be split
    :return: tuple of (list of cropped bounding boxes, list of bounding boxes that could not be cropped)
    """
    cc_bboxes = get_connected_component_bounding_boxes(binary_img)
    potentially_cropped_bboxes = [crop_to_content(bbox, cc_bboxes) for bbox in bboxes]
    if split:
        # store list of cropped bounding boxes (which will have blobs in the binary image)
        cropped_bboxes = [bbox for bbox, cropped in potentially_cropped_bboxes if cropped]
        # store list of non-cropped bounding boxes (those that did not contain any blobs in the binary image)
        non_cropped_bboxes = [bbox for bbox, cropped in potentially_cropped_bboxes if not cropped]
        return cropped_bboxes, non_cropped_bboxes
    else:
        return [bbox for bbox, _ in potentially_cropped_bboxes]


def get_minimal_bounding_boxes_v2(binary_img, bboxes, *, split=False, bboxes_in_lines=False, maintain_center=False):
    """
    Crops the given bounding boxes to contain only the ink, given the binary image.
    Bounding boxes with no ink are left as they are and returned separately.
    Only considers pixels in the bbox to begin with, getting a new connected component list for each bbox
    :param binary_img: black and white image where the black represents ink
    :param bboxes: the bounding boxes to crop to the minimal size while retaining all
    the ink that was fully in the bounding box already
    :param split: if the cropped and non-cropped should be split
    :param bboxes_in_lines: if the bboxes passed in are collected by line
    :param maintain_center: True to keep bounding box centers fixed
    :return: tuple of (list of cropped bounding boxes, list of bounding boxes that could not be cropped)
    """
    if bboxes_in_lines:
        return [get_minimal_bounding_boxes_v2(binary_img, line, split=split,
                                              maintain_center=maintain_center) for line in bboxes]

    _, min_height = get_mean_dims(bboxes)
    potentially_cropped_bboxes = [crop_to_content(bbox,
                                                  get_connected_component_bounding_boxes(bbox.crop(binary_img)),
                                                  allow_partial=True,
                                                  initial_offset=(bbox.x_min, bbox.y_min),
                                                  maintain_center=maintain_center,
                                                  min_height=min_height)
                                  for bbox in bboxes]
    if split:
        # store list of cropped bounding boxes (which will have blobs in the binary image)
        cropped_bboxes = [bbox for bbox, cropped in potentially_cropped_bboxes if cropped]
        # store list of non-cropped bounding boxes (those that did not contain any blobs in the binary image)
        non_cropped_bboxes = [bbox for bbox, cropped in potentially_cropped_bboxes if not cropped]
        return cropped_bboxes, non_cropped_bboxes
    else:
        return [bbox for bbox, _ in potentially_cropped_bboxes]


def get_minimal_line_bounding_boxes(binary_img, lines):
    """
    Crops the given bounding boxes to contain only the ink, given the binary image.
    Bounding boxes with no ink are left as they are and returned separately
    :param binary_img: black and white image where the black represents ink
    :param lines: list of lists of bounding boxes to crop to the minimal size while retaining all
    the ink that was fully in the bounding box already (maintaining centers of line and rough height metrics)
    :return: tuple of (list of cropped bounding boxes, list of bounding boxes that could not be cropped)
    """
    cc_bboxes = get_connected_component_bounding_boxes(binary_img)

    new_lines = []
    for line in lines:
        new_line = []
        for bbox in line:
            _, min_height = get_mean_dims(line)
            cropped_bbox, _ = crop_to_content(bbox, cc_bboxes, maintain_center=True, min_height=min_height)
            new_line.append(cropped_bbox)
        new_lines.append(new_line)
    return new_lines


def crop_to_content(bbox, cc_bboxes, *, min_height=0, maintain_center=False, allow_partial=True,
                    initial_offset=(0, 0)):
    cropped = False
    offset_x, offset_y = initial_offset
    bbox.offset(-offset_x, -offset_y)
    # Crop bbox to the largest region that fits all fully contained glyphs
    # Initialize offset
    x_min, y_min = bbox.x_max, bbox.y_max
    x_max, y_max = bbox.x_min, bbox.y_min
    for cc_bbox in cc_bboxes[1:]:
        if cc_bbox.is_inside(bbox, allow_partial=allow_partial):
            x_min, y_min = min(x_min, cc_bbox.x_min), min(y_min, cc_bbox.y_min)
            x_max, y_max = max(x_max, cc_bbox.x_max), max(y_max, cc_bbox.y_max)
            cropped = True
    if allow_partial:
        x_min, y_min = max(x_min, bbox.x_min), max(y_min, bbox.y_min)
        x_max, y_max = min(x_max, bbox.x_max), min(y_max, bbox.y_max)
    # Undo offset
    # if bbox was cropped, maintain center if needed or make clone of bounding box and return
    if cropped:
        if not maintain_center:
            bbox = BBox(x_min, y_min, x_max, y_max, probabilities=bbox.probabilities, uuid=bbox.uuid)
            bbox.offset(offset_x, offset_y)
        else:
            bbox = recenter_bbox(bbox, x_min, y_min, x_max, y_max)
            bbox.offset(offset_x, offset_y)
        if min_height > 0:
            bbox.grow_to(0, min_height)
    else:
        bbox.offset(offset_x, offset_y)
    return bbox, cropped


def recenter_bbox(bbox, x_min, y_min, x_max, y_max):
    """
    Increases the size of the bounding box region passed into keep the center in the same place, and generates a copy
    of the passed in bounding box with the new dimensions
    :param bbox: bbox to copy and keep center of
    :param x_min:
    :param y_min:
    :param x_max:
    :param y_max:
    :return:
    """
    cx, cy = bbox.center
    # get bigger x dist from center
    max_dx = max(abs(x_min - cx), abs(x_max - cx))
    # get bigger y dist from center
    max_dy = max(abs(y_min - cy), abs(y_max - cy))
    # generate new bounding box with the biggest size that is still within at least two of the bounds passed in
    return BBox(cx - max_dx, cy - max_dy, cx + max_dx, cy + max_dy,
                probabilities=bbox.probabilities, uuid=bbox.uuid)
