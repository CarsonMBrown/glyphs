import numpy as np
import torch

from src.evaluation.bbox_eval import get_unique_bboxes
from src.util.bbox_util import bbox_inside

# Load pre-trained model using given coco data and using yolov5x
model = torch.hub.load('ultralytics/yolov5', 'custom', 'weights/yolov5/mono_raw_xl_300.pt', trust_repo=True)


def find_glyphs(img):
    """
    Gets the bounding boxes of the glyphs in the passed-in image (RGB channels)

    :param img: image to get glyph bounding boxes (image path, cv2 image, or pil image)
    :return:
    """
    return model(img)


def sliding_glyph_window(img, *, window_size=800, window_step=200):
    """
    :param img: img to get bboxes from (IN RGB)
    :param window_size: size of the sliding window to use
    :param window_step: step to take between windows
    :return: returns a list of bounding boxes in pascal form (x_min, y_min, x_max, y_max)
    """
    y_max, x_max, _ = img.shape
    bboxes = []
    for dx in range(0, max(x_max - window_size, 1), window_step):
        for dy in range(0, max(y_max - window_size, 1), window_step):
            # extend window to fit edges of image to prevent narrow or short slices
            window_x_max = x_max if dx + window_step + window_size >= x_max else dx + window_size
            window_y_max = y_max if dy + window_step + window_size >= y_max else dy + window_size
            # generate pascal bounding box inset within window to allow for removal of edge boxes
            valid_bbox_window = (
                dx + window_step if dx != 0 else dx,
                dy + window_step if dy != 0 else dy,
                window_x_max - window_step if window_x_max != x_max else x_max,
                window_y_max - window_step if window_y_max != y_max else y_max
            )
            # get bboxes fully contained by the window
            bboxes += [
                bbox for bbox in
                get_bounding_boxes(find_glyphs(img[dy:window_y_max, dx:window_x_max]), offset_x=dx, offset_y=dy)
                if bbox_inside(bbox, valid_bbox_window)
            ]

    return get_unique_bboxes(bboxes)


def show_result(result):
    result.show()


def results_to_list(result):
    return result.pandas().xyxy[0].values


def get_bounding_boxes(result, *, offset_x=0, offset_y=0):
    """returns a list of bounding boxes in pascal form (x_min, y_min, x_max, y_max)"""
    return [tuple(np.array([min_x + offset_x, min_y + offset_y, max_x + offset_x, max_y + offset_y], int))
            for min_x, min_y, max_x, max_y, _, _, _ in results_to_list(result)]


def get_bounding_box_confidence(result):
    return [confidence
            for _, _, _, _, confidence, _, _ in results_to_list(result)]
