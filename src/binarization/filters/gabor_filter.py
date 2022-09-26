import cv2
import numpy as np

from src.util.dir_util import get_input_images, set_output_dir
from src.util.img_util import load_image, save_image


def build_filters():
    """

    Author: https://github.com/kkomakkoma
    :return:
    """
    filters = []
    ksize = 31
    for theta in np.arange(0, np.pi, np.pi / 2):
        params = {'ksize': (ksize, ksize), 'sigma': 0.5, 'theta': theta, 'lambd': 5.0,
                  'gamma': 0.04, 'psi': 0, 'ktype': cv2.CV_32F}
        kern = cv2.getGaborKernel(**params)
        kern /= 1.5 * kern.sum()
        filters.append((kern, params))
    return filters


def process(img_in_dir, img_out_dir, filters=None):
    """

    Author: https://github.com/kkomakkoma
    :param img_in_dir: path to directory containing input images
    :param img_out_dir: path to directory to output images
    :param filters:
    :return:
    """
    img_list = get_input_images(img_in_dir)
    set_output_dir(img_out_dir)

    if filters is None:
        filters = build_filters()

    for img_path in img_list:
        img, img_output = load_image(img_in_dir, img_out_dir, img_path, "gabor", gray_scale=True)
        results = []
        for kern, _ in filters:
            fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
            results.append(fimg)
        final_image = cv2.addWeighted(results[0], 0.5, results[1], 0.5, 0)
        inv_weight = 2
        for r in results[2:]:
            inv_weight += 1
            weight = 1 / inv_weight
            final_image = cv2.addWeighted(final_image, 1 - weight, r, weight, 0)
        save_image(img_output, final_image)
