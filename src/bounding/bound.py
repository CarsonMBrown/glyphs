from src.bounding.connected_components import get_connected_component_bounding_boxes, bound_and_render
from src.util.bbox_util import BBox
from src.util.line_util import get_mean_line_dims


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

    for line in lines:
        mean_width, mean_height = get_mean_line_dims(line)
        for bbox in line:
            cropped_bbox, cropped = crop_to_content(bbox, cc_bboxes, maintain_center=True)
            if bbox.height < mean_height:
                bbox.set_height(mean_height)
            if not cropped:
                continue
    return lines


def crop_to_content(bbox, cc_bboxes, *, maintain_center=False):
    cropped = False
    # Crop bbox to the largest region that fits all fully contained glyphs
    x_min, y_min, x_max, y_max = bbox.x_max, bbox.y_max, bbox.x_min, bbox.y_min
    for cc_bbox in cc_bboxes:
        if cc_bbox.is_inside(bbox):
            x_min, y_min = min(x_min, cc_bbox.x_min), min(x_min, cc_bbox.y_min)
            x_max, y_max = max(x_max, cc_bbox.x_max), max(x_max, cc_bbox.y_max)
            cropped = True
    if cropped:
        if not maintain_center:
            bbox = BBox(x_min, y_min, x_max, y_max, probabilities=bbox.probabilities, uuid=bbox.uuid)
        else:
            bbox = recenter_bbox(bbox, x_max, x_min, y_max, y_min)
    return bbox, cropped


def recenter_bbox(bbox, x_max, x_min, y_max, y_min):
    cx, cy = bbox.center
    max_dx = max(abs(x_min - cx), abs(x_max - cx))
    max_dy = max(abs(y_min - cy), abs(y_max - cy))
    return BBox(cx - max_dx, cy - max_dy, cx + max_dx, cy + max_dy,
                probabilities=bbox.probabilities, uuid=bbox.uuid)
