import cv2
import numpy as np

from src.util.dir_util import get_input_images, set_output_dir
from src.util.img_util import load_image, save_image

BOUNDING_COLOR = [0, 0, 255]
CENTROID_COLOR = [0, 255, 0]


def bound(img_in_dir, img_out_dir):
    """
    Takes in a binararized image and returns

    Author: https://www.geeksforgeeks.org/python-opencv-connected-component-labeling-and-analysis/
    :param img_in_dir: path to directory containing input images
    :param img_out_dir: path to directory to output images
    :return:
    """
    img_list = get_input_images(img_in_dir)
    set_output_dir(img_out_dir)
    bounding_boxs = []

    for img_path in img_list:
        img, img_output = load_image(img_in_dir, img_out_dir, img_path, "connected", invert=True)

        # Apply the Component analysis function
        analysis = cv2.connectedComponentsWithStats(img, 8, cv2.CV_32S)
        (totalLabels, label_ids, values, centroids) = analysis

        color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        imgY, imgX = np.shape(img)

        # Merge bounding boxes with centroids in other boxes or that are fully in other boxs
        internal_centroids = get_internal_centroids(centroids, values)
        internal_regions = get_internal_regions(values)

        for i, (X, Y, W, H, A) in enumerate(values[1:], 1):
            # skip bounding boxs with centroids in other boxs
            if i in internal_regions or i in internal_centroids:
                continue

            bounding_boxs.append(tuple(X, Y, W, H, A))

            # Draw centroid
            centroid = np.array(np.floor(centroids[i]), int)
            radius = 1
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    color_img[centroid[1] + dy][centroid[0] + dx] = CENTROID_COLOR

            # Draw bounding boxs
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
        return bounding_boxs


def get_internal_centroids(centroids, values):
    """
    :param centroids:
    :param values:
    :return: a list of the indexes/labels of all
    the bounding boxs with centroids that are fully in another bounding box
    """
    to_remove = []
    for i, (cx, cy) in enumerate(centroids[1:], 1):
        for x, y, w, h, _ in values[1:i]:
            if x <= cx <= x + w and y <= cy <= y + h:
                to_remove.append(i)
    return to_remove


def get_internal_regions(values):
    """
    :param values:
    :return: a list of the indexes/labels of all
    the bounding boxs that are fully in another bounding box
    """
    to_remove = []
    for i, (ax, ay, aw, ah, _) in enumerate(values[1:], 1):
        ax2, ay2 = ax + aw, ay + ah
        for bx, by, bw, bh, _ in values[1:i]:
            if ax >= bx and ay >= by and ax2 <= bx + bw and ay2 <= by + bh:
                to_remove.append(i)
    return to_remove
