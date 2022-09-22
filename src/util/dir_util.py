import os


def get_input_images(img_in_dir):
    # SET DIRECTORY FOR INPUT
    print("Image input directory:", img_in_dir)
    # GET ALL IMAGES
    img_list = [i for i in os.listdir(img_in_dir)
                if not os.path.isdir(os.path.join(img_in_dir, i))]
    return img_list


def set_output_dir(img_out_dir):
    # SET DIRECTORY FOR OUTPUT
    if not os.path.exists(img_out_dir):
        os.makedirs(img_out_dir)
    print("Image output directory:", img_out_dir)
