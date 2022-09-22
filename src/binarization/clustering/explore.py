from PIL import ImageDraw
from matplotlib import pyplot as plt

from ICFHR2022 import *
from imageutil import *
from scaledimage import ScaledImage

data = ICFHR2022()


def make_palette(colors):
    pal = Image.new('RGB', (60, 20 * len(colors)), color='white')
    draw = ImageDraw.Draw(pal)
    for i in range(len(colors)):
        draw.rectangle([0, i * 20, 60, (i + 1) * 20], fill=colors[i])
    return pal


MAX_IMG_SIZE = 1000


def find_colors(img_data):
    _, mult = plt.subplots(2, 3, figsize=(10, 5))
    for img_id in img_data.img_ids():
        im = img_data.id_to_img[img_id]
        f = img_file(im)
        img = Image.open(dir + f)
        scaled = ScaledImage(img, MAX_IMG_SIZE)
        cutout = cut_out(scaled.img, 0.4, 0.4)
        centers = cluster_colours(cutout, 6)
        mult[0][0].imshow(scaled.img)
        mult[0][1].imshow(colour_center(scaled.img, centers, ['black', None, None, None, None, None], 'white'))
        mult[0][2].imshow(make_palette(centers))

        blurred = apply_blur(scaled.img)
        cutout = cut_out(blurred, 0.4, 0.4)
        centers = cluster_colours(cutout, 4)
        mult[1][0].imshow(blurred)
        mult[1][1].imshow(colour_center(blurred, centers, ['white', None, None, None], 'black'))
        mult[1][2].imshow(find_sobel(blurred), cmap='Greys')
        break


def find_image_sizes(data):
    for id in data.img_ids():
        im = data.id_to_img[id]
        f = img_file(im)
        img = Image.open(dir + f)
        print(img.size)


find_colors(data)
