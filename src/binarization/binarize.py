from src.binarization.clustering.cluster import kmeans
from src.binarization.dp_linknet import dp_linknet
from src.binarization.filters import gabor_filter
from src.binarization.thresholding import input_sensitive_threshold
from src.util.dir_util import clean_dir, get_input_img_paths, init_output_dir
from src.util.img_util import load_image, save_image


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
    :param img_temp_dir: path to temp directory that will be cleared
    :param img_out_dir: path to directory to output images
    :return: None
    """
    gabor_filter.process(img_in_dir, img_temp_dir)
    dp_linknet.binarize_imgs(img_temp_dir, img_out_dir, threshold=5)
    clean_dir(img_temp_dir)


def sensitive_threshold(img_in_dir, img_out_dir):
    """
    :param img_in_dir: path to directory containing input images
    :param img_out_dir: path to directory to output images
    :return: None
    """
    input_sensitive_threshold.run(img_in_dir, img_out_dir)


def cluster(img_in_dir, img_out_dir):
    """
    :param img_in_dir: path to directory containing input images
    :param img_out_dir: path to directory to output images
    :return: None
    """
    kmeans(img_in_dir, img_out_dir)


def cnn(img_in_dir, img_out_dir, *, threshold=5):
    """
    :param img_in_dir: path to directory containing input images
    :param img_out_dir: path to directory to output images
    :param threshold:
    :return: None
    """
    dp_linknet.binarize_imgs(img_in_dir, img_out_dir, threshold=threshold)


def mask(img_in_dir, mask_in_dir, img_out_dir, *, invert_mask=True, removed_color=255):
    img_lists = zip(get_input_img_paths(img_in_dir), get_input_img_paths(mask_in_dir))
    init_output_dir(img_out_dir)

    for img_path, mask_path in img_lists:
        img, img_output = load_image(img_in_dir, img_out_dir, img_path, gray_scale=True, skip_existing=False)
        mask_img, _ = load_image(mask_in_dir, img_out_dir, mask_path, gray_scale=True, invert=invert_mask,
                                 skip_existing=False)
        if img is None or mask_img is None:
            continue
        if img.shape != mask_img.shape:
            continue
        Y, X = img.shape
        for y in range(Y):
            for x in range(X):
                if mask_img[y, x] == 255 and img[y, x] == 0:
                    img[y, x] = removed_color

        save_image(img_output, img)
