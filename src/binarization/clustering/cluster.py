from math import dist

import cv2
from sklearn.cluster import KMeans

from src.util.dir_util import get_input_img_paths, set_output_dir
from src.util.img_util import load_image, save_image


def kmeans(img_in_dir, img_out_dir, k=5):
    """
    Uses kmeans clustering on pixel values to generate a binarization of the image
    :param img_in_dir: glyph_path to directory containing input images
    :param img_out_dir: glyph_path to directory to output images
    :return: None
    """
    img_list = get_input_img_paths(img_in_dir)
    set_output_dir(img_out_dir)

    for img_path in img_list:
        img, img_output = load_image(img_in_dir, img_out_dir, img_path)
        if img is None:
            continue
        pixels = img.reshape((-1, 3))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        shape = img.shape
        k_means = KMeans(n_clusters=k)
        k_means.fit(pixels)
        labels, centroids = k_means.labels_, k_means.cluster_centers_
        black = [0, 0, 0]
        current_c = [255, 255, 255]
        darkest_index = -1
        for i, c in enumerate(centroids):
            if dist(current_c, black) < dist(c, black):
                current_c = c
                darkest_index = i

        for y in range(shape[0]):
            for x in range(shape[1]):
                img[y][x] = 255 if labels[(y * shape[1]) + x] else 0

        save_image(img_output, img)
