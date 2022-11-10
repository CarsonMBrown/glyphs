from src.binarization.dp_linknet import dp_linknet
from src.binarization.filters import gabor_filter
from src.binarization.thresholding import input_sensitive_threshold
from src.util.dir_util import clean_dir


def gabor(img_in_dir, img_out_dir):
    """
    :param img_in_dir: path to directory containing input images
    :param img_out_dir: path to directory to output images
    :return: None
    """
    gabor_filter.process(img_in_dir, img_out_dir)


def gabor_inverse(img_in_dir, img_temp_dir, img_out_dir):
    """
    Uses the gabor filter and cnn binarization to
    generate a binary image with black (non-character) elements.

    :param img_in_dir: path to directory containing input images
    :param img_out_dir: path to directory to output images
    :return: None
    """
    gabor_filter.process(img_in_dir, img_temp_dir)
    test.binarize_imgs(img_temp_dir, img_out_dir)
    clean_dir(img_temp_dir)


def sensitive_threshold(img_in_dir, img_out_dir):
    """
    :param img_in_dir: path to directory containing input images
    :param img_out_dir: path to directory to output images
    :return: None
    """
    input_sensitive_threshold.run(img_in_dir, img_out_dir)


def clustering(img_in_dir, img_out_dir):
    """
    :param img_in_dir: path to directory containing input images
    :param img_out_dir: path to directory to output images
    :return: None
    """
    # TODO
    pass


def cnn(img_in_dir, img_out_dir, threshold=5):
    """
    :param img_in_dir: path to directory containing input images
    :param img_out_dir: path to directory to output images
    :return: None
    """
    dp_linknet.binarize_imgs(img_in_dir, img_out_dir, threshold=threshold)