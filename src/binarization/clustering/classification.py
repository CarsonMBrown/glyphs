import math

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from ICFHR2022 import *
from imageutil import *
from scaledimage import ScaledImage

data = ICFHR2022()

MAX_IMG_SIZE = 1000


def find_tokens(name, base, foot):
    # _, mult = plt.subplots(1, 2, figsize = (10,5))
    good_tokens = []
    for id in data.img_ids()[0:20]:
        print('-----------------', id)
        im = data.id_to_img[id]
        f = img_file(im)
        img = Image.open(dir + f)
        scaled = ScaledImage(img, MAX_IMG_SIZE)
        centers = cluster_colours(scaled.img, 5)
        anns = data.img_to_anns[id]
        n_from_image = 0
        for a in anns:
            (cl_name, cl_base, cl_foot) = data.ann_class(a)
            if cl_name == name and n_from_image < 2:
                part = img.crop((ann_x(a), ann_y(a), ann_x(a) + ann_w(a), ann_y(a) + ann_h(a)))
                colored = colour_center(part, centers, ['black', 'white', 'white', 'white', 'white'])
                full = fullness(colored)
                print(full)
                if full > 0.20:
                    text = cl_base + " " + cl_foot
                    good_tokens.append((colored, text))
                    n_from_image = n_from_image + 1
                # plt.figure(1)
                # mult[0].imshow(part)
                # mult[1].imshow(colored)
                # mult[1].set_title(text)
                # plt.imshow(colored)
                # plt.pause(1)
    size = max(1, math.ceil(math.sqrt(len(good_tokens))))
    _, axs = plt.subplots(size, size, figsize=(15, 10), squeeze=False)
    for i in range(len(good_tokens)):
        row = math.floor(i / size)
        col = i % size
        axs[row, col].imshow(good_tokens[i][0])
        axs[row, col].set_ylabel(good_tokens[i][1])
    plt.pause(200)


find_tokens('Α', 'bt1', 'ft1')
