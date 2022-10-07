from src.bounding.connected_components import connected_components


def connected_component(img_in_dir, img_out_dir):
    """
    :param img_in_dir: glyph_path to directory containing input images
    :param img_out_dir: glyph_path to directory to output images
    :return: None
    """
    print(connected_components.bound(img_in_dir, img_out_dir))
