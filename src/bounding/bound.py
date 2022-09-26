from src.bounding.connected_components import connected_components


def connected_component(img_in_dir, img_out_dir):
    """
    :param img_in_dir: path to directory containing input images
    :param img_out_dir: path to directory to output images
    :return: None
    """
    connected_components.bound(img_in_dir, img_out_dir)
