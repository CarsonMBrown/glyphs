from src.bounding.connected_components import get_connected_component_bounding_boxes, bound_and_render
from src.util.bbox_util import BBox


def export_connected_component(img_in_dir, img_out_dir):
    """
    Bound using connected components and then save images
    :param img_in_dir: glyph_path to directory containing input images
    :param img_out_dir: glyph_path to directory to output images
    :return: None
    """
    print(bound_and_render(img_in_dir, img_out_dir))


def get_minimal_bounding_boxes(binary_img, bboxes):
    """
    Crops the given bounding boxes to contain only the ink, given the binary image.
    Bounding boxes with no ink are left as they are and returned separately
    :param binary_img: black and white image where the black represents ink
    :param bboxes: the bounding boxes to crop to the minimal size while retaining all
    the ink that was fully in the bounding box already
    :return: tuple of (list of cropped bounding boxes, list of bounding boxes that could not be cropped)
    """
    cc_bboxes = get_connected_component_bounding_boxes(binary_img)
    potentially_cropped_bboxes = [crop_to_content(bbox, cc_bboxes) for bbox in bboxes]
    # store list of cropped bounding boxes (which will have blobs in the binary image)
    cropped_bboxes = [bbox for bbox, cropped in potentially_cropped_bboxes if cropped]
    # store list of non-cropped bounding boxes (those that did not contain any blobs in the binary image)
    non_cropped_bboxes = [bbox for bbox, cropped in potentially_cropped_bboxes if not cropped]
    return cropped_bboxes, non_cropped_bboxes


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
            cropped_bbox, _ = crop_to_content(bbox, cc_bboxes, maintain_center=True)
            new_line.append(cropped_bbox)
        new_lines.append(new_line)
    return new_lines


def crop_to_content(bbox, cc_bboxes, *, maintain_center=False):
    cropped = False
    # Crop bbox to the largest region that fits all fully contained glyphs
    x_min, y_min, x_max, y_max = bbox.x_max, bbox.y_max, bbox.x_min, bbox.y_min
    for cc_bbox in cc_bboxes:
        if cc_bbox.is_inside(bbox):
            x_min, y_min = min(x_min, cc_bbox.x_min), min(x_min, cc_bbox.y_min)
            x_max, y_max = max(x_max, cc_bbox.x_max), max(x_max, cc_bbox.y_max)
            cropped = True
    # if bbox was cropped, maintain center if needed or make clone of bounding box and return
    if cropped:
        if not maintain_center:
            bbox = BBox(x_min, y_min, x_max, y_max, probabilities=bbox.probabilities, uuid=bbox.uuid)
        else:
            bbox = recenter_bbox(bbox, x_max, x_min, y_max, y_min)
    return bbox, cropped


def recenter_bbox(bbox, x_max, x_min, y_max, y_min):
    """
    Increases the size of the bounding box region passed into keep the center in the same place, and generates a copy
    of the passed in bounding box with the new dimensions
    :param bbox: bbox to copy and keep center of
    :param x_max:
    :param x_min:
    :param y_max:
    :param y_min:
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
