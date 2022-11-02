import os
import pickle
import random
from math import ceil

import cv2


def split_eval_data(train_img_dir, train_label_dir, eval_img_dir, eval_label_dir, eval_percentage=.3, seed=0):
    """
    Given a directory containing train images, extracts some of those images to an eval directory
    :param train_img_dir:
    :param train_label_dir:
    :param eval_img_dir:
    :param eval_label_dir:
    :param eval_percentage:
    :param seed:
    :return:
    """
    random.seed(seed)
    imgs = get_input_img_paths(train_img_dir)
    random.shuffle(imgs)
    for i in imgs[:ceil(len(imgs) * eval_percentage)]:
        img_name, img_extension = split_image_name_extension(i)
        os.rename(os.path.join(train_img_dir, img_name + img_extension),
                  os.path.join(eval_img_dir, img_name + img_extension))
        os.rename(os.path.join(train_label_dir, img_name + ".txt"),
                  os.path.join(eval_label_dir, img_name + ".txt"))


def get_input_img_paths(img_in_dir, *, by_dir=False, verbose=True):
    # set directory for input
    if verbose:
        print("Image input directory:", img_in_dir)
    if not by_dir:
        img_files = []
        # get all image files, searching child dirs as well
        for dir_path, dirs, files in os.walk(img_in_dir):
            for filename in files:
                file_path = os.path.join(dir_path, filename)
                if file_path.lower().endswith('.png') or file_path.lower().endswith(".jpg"):
                    img_files.append(file_path.removeprefix(img_in_dir)[1:])
        return img_files
    # else return images in dict with dir name as key to list
    categories = [i for i in os.listdir(img_in_dir)
                  if os.path.isdir(os.path.join(img_in_dir, i))]
    category_dict = {}
    for c in categories:
        category_dict[c] = [os.path.join(img_in_dir, c, p) for p in
                            get_input_img_paths(os.path.join(img_in_dir, c), verbose=False)]
    return category_dict


IMAGE_LIST_CACHE_FILE_NAME = "img_list.pickle"
BY_DIR_IMAGE_CACHE_FILE_NAME = "img_dict.pickle"


def get_input_imgs(img_in_dir, *, by_dir=False, verbose=True, cache=True):
    paths = get_input_img_paths(img_in_dir, by_dir=by_dir, verbose=verbose)
    if by_dir:
        cache_path = os.path.join(img_in_dir, BY_DIR_IMAGE_CACHE_FILE_NAME)
        if os.path.exists(cache_path):
            with open(cache_path, mode="rb") as f:
                return pickle.load(f)
        img_dict = {}
        for c in paths.keys():
            img_dict[c] = [cv2.imread(img_path) for img_path in paths[c]]
        if cache:
            with open(cache_path, mode="wb") as f:
                pickle.dump(img_dict, f)
        return img_dict
    else:
        cache_path = os.path.join(img_in_dir, IMAGE_LIST_CACHE_FILE_NAME)
        if os.path.exists(cache_path):
            with open(cache_path, mode="rb") as f:
                return pickle.load(f)
        img_list = [cv2.imread(os.path.join(img_in_dir, img_path)) for img_path in paths]
        if cache:
            with open(cache_path, mode="wb") as f:
                pickle.dump(img_list, f)
        return img_list


def set_output_dir(img_out_dir):
    # SET DIRECTORY FOR OUTPUT
    if not os.path.exists(img_out_dir):
        os.makedirs(img_out_dir)
    print("Image output directory:", img_out_dir)


def clean_dir(temp_dir):
    img_list = get_input_img_paths(temp_dir)
    for i in img_list:
        os.remove(os.path.join(temp_dir, i))
    print("Cleared directory:", temp_dir)


def get_file_name(img_path):
    if "/" in img_path:
        return img_path[img_path.rindex("/") + 1:img_path.rindex(".")]
    return img_path[:img_path.rindex(".")]


def split_image_name_extension(img_path):
    return get_file_name(img_path), img_path[img_path.rindex("."):]
