from src.util.dir_util import *
from src.util.img_util import *


@Warning
def run(img_in_dir, img_out_dir):
    """
    Runs algorithm from "Input sensitive thresholding for ancient Hebrew manuscript" by Itay Bar-Yosef
    :param img_in_dir: glyph_path to directory containing input images
    :param img_out_dir: glyph_path to directory to output images
    :return: None
    """
    img_list = get_input_img_paths(img_in_dir)
    set_output_dir(img_out_dir)

    for img_path in img_list:
        img, img_output = load_image(img_in_dir, img_out_dir, img_path, gray_scale=True)
        # BINARIZE USING THRESHOLD
        _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
        save_image(img_output, img)
