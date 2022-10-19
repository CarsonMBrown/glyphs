import os


def get_input_images(img_in_dir, *, by_dir=False, verbose=True):
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
                            get_input_images(os.path.join(img_in_dir, c), verbose=False)]
    return category_dict


def set_output_dir(img_out_dir):
    # SET DIRECTORY FOR OUTPUT
    if not os.path.exists(img_out_dir):
        os.makedirs(img_out_dir)
    print("Image output directory:", img_out_dir)


def clean_dir(temp_dir):
    img_list = get_input_images(temp_dir)
    for i in img_list:
        os.remove(os.path.join(temp_dir, i))
    print("Cleared directory:", temp_dir)
