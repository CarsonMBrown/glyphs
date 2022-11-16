import cv2
import numpy as np

from src.util.bbox_util import BBox
from src.util.dir_util import get_input_img_paths, set_output_dir
from src.util.img_util import load_image, save_image

BOUNDING_COLOR = [0, 0, 255]
CENTROID_COLOR = [0, 255, 0]


def get_connected_component_bounding_boxes(img):
    # Invert img so black is non-ink, white is ink
    inverted_img = cv2.invert(img)
    # Apply the Component analysis function
    _, _, values, _ = cv2.connectedComponentsWithStats(inverted_img, 8, cv2.CV_32S)
    # convert bounding boxes to BBoxes and return
    return [BBox.from_coco(x, y, w, h) for x, y, w, h, _ in values]


def bound_and_render(img_in_dir, img_out_dir):
    """
    Using connected components, draws a bounding box around each non-internal component and saves the results to file.

    Author: https://www.geeksforgeeks.org/python-opencv-connected-component-labeling-and-analysis/
    :param img_in_dir: glyph_path to directory containing input images
    :param img_out_dir: glyph_path to directory to output images
    :return:
    """
    img_list = get_input_img_paths(img_in_dir)
    set_output_dir(img_out_dir)
    bounding_boxes = []

    for img_path in img_list:
        img, img_output = load_image(img_in_dir, img_out_dir, img_path, invert=True)

        # Apply the Component analysis function
        analysis = cv2.connectedComponentsWithStats(img, 8, cv2.CV_32S)
        (totalLabels, label_ids, values, centroids) = analysis

        color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        imgY, imgX = np.shape(img)

        # Merge bounding boxes with centroids in other boxes or that are fully in other boxes
        internal_centroids = get_internal_centroids(centroids, values)
        internal_regions = get_internal_regions(values)

        for i, (X, Y, W, H, A) in enumerate(values[1:], 1):
            # skip bounding boxes with centroids in other boxes
            if i in internal_regions or i in internal_centroids:
                continue

            bounding_boxes.append((X, Y, W, H, A))

            # Draw centroid
            centroid = np.array(np.floor(centroids[i]), int)
            radius = 1
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    color_img[centroid[1] + dy][centroid[0] + dx] = CENTROID_COLOR

            # Draw bounding boxes
            # outset coords by 1 to not be inclusive of edges
            X, Y = max(X - 1, 0), max(Y - 1, 0)
            W, H = min(imgX - X - 1, W + 2), min(imgY - Y - 1, H + 2)
            # Draw horizontal lines
            for dx in range(W):
                color_img[Y][X + dx] = BOUNDING_COLOR
                color_img[Y + H - 1][X + dx] = BOUNDING_COLOR
            # Draw vertical lines
            for dy in range(H):
                color_img[Y + dy][X] = BOUNDING_COLOR
                color_img[Y + dy][X + W - 1] = BOUNDING_COLOR

        save_image(img_output, color_img)
        return bounding_boxes


def get_internal_centroids(centroids, values):
    """
    Get a list of all the bounding boxes with centroids inside at least one other bounding box
    :param centroids:
    :param values:
    :return: a list of the indexes/labels of all the bounding boxes with
    centroids that are fully in another bounding box
    """
    to_remove = []
    for i, (cx, cy) in enumerate(centroids[1:], 1):
        for x, y, w, h, _ in values[1:i]:
            if x <= cx <= x + w and y <= cy <= y + h:
                to_remove.append(i)
    return to_remove


def get_internal_regions(values):
    """
    Get list of all the bounding boxes that are fully contained within another bounding box
    :param values:
    :return: a list of the indexes/labels of all the bounding boxes that are fully in another bounding box
    """
    internal_bboxes = []
    for i, (ax, ay, aw, ah, _) in enumerate(values[1:], 1):
        ax2, ay2 = ax + aw, ay + ah
        for bx, by, bw, bh, _ in values[1:i]:
            if ax >= bx and ay >= by and ax2 <= bx + bw and ay2 <= by + bh:
                internal_bboxes.append(i)
    return internal_bboxes
