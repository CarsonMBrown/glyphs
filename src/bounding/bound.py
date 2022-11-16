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


def cropped_bounding_boxes(binary_img, bboxes):
    """
    Crops the given bounding boxes to contain only the ink, given the binary image.
    Bounding boxes with no ink are left as they are and returned separately
    :param binary_img: black and white image where the black represents ink
    :param bboxes: the bounding boxes to crop to the minimal size while retaining all
    the ink that was fully in the bounding box already
    :return: tuple of (list of cropped bounding boxes, list of bounding boxes that could not be cropped)
    """
    cc_bboxes = get_connected_component_bounding_boxes(binary_img)
    cropped_bboxes = []
    non_cropped_bboxes = []

    for bbox in bboxes:
        cropped = False
        # Crop bbox to the largest region that fits all fully contained glyphs
        x_min, y_min, x_max, y_max = bbox.x_max, bbox.y_max, bbox.x_min, bbox.y_min
        for cc_bbox in cc_bboxes:
            if cc_bbox.is_inside(bbox):
                x_min, y_min = min(x_min, cc_bbox.x_min), min(x_min, cc_bbox.y_min)
                x_max, y_max = max(x_max, cc_bbox.x_max), max(x_max, cc_bbox.y_max)
                cropped = True

        # add bbox to correct list
        if cropped:
            cropped_bboxes.append(BBox(x_min, y_min, x_max, y_max))
        else:
            non_cropped_bboxes.append(bbox)

    return cropped_bboxes, non_cropped_bboxes
