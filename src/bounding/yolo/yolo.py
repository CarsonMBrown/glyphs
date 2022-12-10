import os.path

import cv2
import torch

from src.evaluation.bbox_eval import get_non_enclosed_bboxes
from src.util.bbox import BBox
from src.util.img_util import plot_bboxes

# don't load model until needed
model = None

cache = {}


def find_glyphs(img, *, model_local=True):
    """
    Gets the bounding boxes of the glyphs in the passed-in image (RGB channels)

    :param img: image to get glyph bounding boxes (image path, cv2 image, or pil image)
    :param model_local: if model is not already loaded, load model from local path instead of github
    :return:
    """
    global model
    if model is None:
        if model_local:
            model = torch.hub.load(r'C:\yolov5', 'custom', 'weights/yolov5/mono_raw_xl_300.pt',
                                   source='local')
        else:
            model = torch.hub.load('ultralytics/yolov5', 'custom', 'weights/yolov5/mono_raw_xl_300.pt',
                                   trust_repo=True)

    return model(img)


# TODO: HYPER PARAM TUNING ON WINDOW SIZE AND STEP
def sliding_glyph_window(img, *, window_size=800, window_step=200, bbox_gif_export_path=None,
                         inner_sliding_window=True, confidence_min=None, cache_name=None, duplicate_threshold=.8):
    """
    :param img: img to get bboxes from (IN BGR)
    :param window_size: size of the sliding window to use
    :param window_step: step to take between windows
    :param bbox_gif_export_path: if not None, the draws each sliding window to this path as an image
    :param inner_sliding_window: if a smaller inner window should be used
    :param confidence_min: min confidence to keep bboxes
    :param cache_name: where to save bboxes in memory, None to not save
    :param duplicate_threshold: IOU threshold to not add a bbox
    :return: returns a list of bounding boxes
    """
    if bbox_gif_export_path is not None:
        os.makedirs(bbox_gif_export_path, exist_ok=True)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    y_max, x_max, _ = img.shape
    bboxes = []
    for dx in range(0, max(x_max - window_size, 1), window_step):
        for dy in range(0, max(y_max - window_size, 1), window_step):
            # extend window to fit edges of image to prevent narrow or short slices
            window_x_max = x_max if dx + window_step + window_size >= x_max else dx + window_size
            window_y_max = y_max if dy + window_step + window_size >= y_max else dy + window_size

            # get bboxes fully contained by the window
            if cache_name is not None and str(f"{cache_name}{dy}{dx}") in cache:
                found_glyphs = cache[f"{cache_name}{dy}{dx}"]
            else:
                found_glyphs = find_glyphs(img[dy:window_y_max, dx:window_x_max])
                if cache_name is not None:
                    cache[f"{cache_name}{dy}{dx}"] = found_glyphs

            potential_bboxes = get_bounding_boxes(found_glyphs,
                                                  offset_x=dx,
                                                  offset_y=dy,
                                                  confidence_min=confidence_min)
            # init valid bbox so it can be drawn if needed
            valid_bbox_window = BBox(0, 0, 0, 0)
            if inner_sliding_window:
                # generate bounding box inset within window to allow for removal of edge boxes
                valid_bbox_window = BBox(
                    dx + window_step if dx != 0 else 0,
                    dy + window_step if dy != 0 else 0,
                    window_x_max - window_step if window_x_max != x_max else x_max,
                    window_y_max - window_step if window_y_max != y_max else y_max
                )
                valid_bboxes = [bbox for bbox in potential_bboxes if bbox.is_inside(valid_bbox_window)]
                bboxes += valid_bboxes
            else:
                border = int(window_size * .01)
                # remove boxes that touch edge
                valid_bbox_window = BBox(
                    dx + border if dx != 0 else border,
                    dy + border if dy != 0 else border,
                    window_x_max - border if window_x_max != x_max else x_max - border,
                    window_y_max - border if window_y_max != y_max else y_max - border
                )
                valid_bboxes = [bbox for bbox in potential_bboxes if bbox.is_inside(valid_bbox_window)]
                unique_bboxes = []
                for bbox in valid_bboxes:
                    pair, iou = bbox.get_pair(bboxes)
                    if iou < duplicate_threshold:
                        unique_bboxes.append(bbox)
                    else:
                        if bbox.confidence > pair.confidence:
                            bboxes.remove(pair)
                            unique_bboxes.append(bbox)
                bboxes += unique_bboxes

            # if exporting images
            if bbox_gif_export_path is not None:
                window = BBox(dx, dy, window_x_max, window_y_max)
                temp_img = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR)

                plot_bboxes(temp_img, [window], color=(255, 0, 0), wait=None)
                if inner_sliding_window:
                    plot_bboxes(temp_img, [valid_bbox_window], color=(0, 255, 0), wait=None)
                plot_bboxes(temp_img, potential_bboxes, color=(0, 0, 255), wait=None)
                plot_bboxes(temp_img, bboxes, color=(0, 0, 0), wait=None)
                cv2.imwrite(os.path.join(bbox_gif_export_path, f"window_{dx}_{dy}.png"), temp_img)

    non_internal_bboxes = get_non_enclosed_bboxes(bboxes)
    return non_internal_bboxes


def show_result(result):
    result.show()


def results_to_list(result):
    return result.pandas().xyxy[0].values


# .6075 best based on yolo results and fscore does not increase significantly with higher
def get_bounding_boxes(result, *, offset_x=0, offset_y=0, confidence_min=0.6075):
    if confidence_min is None:
        confidence_min = 0.513
    """returns a list of bounding boxes"""
    return [BBox(min_x + offset_x, min_y + offset_y, max_x + offset_x, max_y + offset_y, confidence=confidence)
            for min_x, min_y, max_x, max_y, confidence, _, _ in results_to_list(result)
            if confidence > confidence_min]


def get_bounding_box_confidence(result):
    return [confidence
            for _, _, _, _, confidence, _, _ in results_to_list(result)]
