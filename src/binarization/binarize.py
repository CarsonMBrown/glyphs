from src.binarization.dp_linknet import test
from src.binarization.filters import gabor_filter
from src.binarization.thresholding import input_sensitive_threshold


def gabor(image_path, output_path):
    gabor_filter.process(image_path, output_path)


def sensitive_threshold(image_path, output_path):
    input_sensitive_threshold.run(image_path, output_path)


def clustering(image_path, output_path):
    pass


def cnn(img_in_dir, img_out_dir):
    """
    :param img_in_dir: path to directory containing input images
    :param img_out_dir: path to directory to output images
    :return: None
    """
    test.run(img_in_dir, img_out_dir)
