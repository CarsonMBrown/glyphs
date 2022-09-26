import math

import cv2
import numpy as np
import skimage.exposure as exposure
from PIL import ImageFile, Image, ImageColor
from sklearn.cluster import KMeans

ImageFile.LOAD_TRUNCATED_IMAGES = True


def gray_of_pixel(p):
    return p[0] + p[1] + p[2]


def weighted_average(v1, v2, w):
    return (v1 + w * v2) / (1 + w)


def average_pixel_weighted(p1, p2, w):
    return (weighted_average(p1[0], p2[0], w),
            weighted_average(p1[1], p2[1], w),
            weighted_average(p1[2], p2[2], w))


def lighter_pixel(p1, p2):
    return gray_of_pixel(p1) > gray_of_pixel(p2)


def int_pixel(p):
    return (math.floor(p[0]), math.floor(p[1]), math.floor(p[2]))


def pixel_dist(p1, p2):
    return math.sqrt((p1[0] - p2[0]) * (p1[0] - p2[0]) +
                     (p1[1] - p2[1]) * (p1[1] - p2[1]) +
                     (p1[2] - p2[2]) * (p1[2] - p2[2]))


def cluster_colours(img, n):
    (w, h) = img.size
    vals = []
    for x in range(w):
        for y in range(h):
            vals.append(img.getpixel((x, y)))
    kmeans = KMeans(n_clusters=n, n_init=3, max_iter=10).fit(vals)
    clusters = sorted(kmeans.cluster_centers_, key=lambda p: gray_of_pixel(p))
    return [int_pixel(p) for p in clusters]


def closest(centers, pixel):
    return sorted(range(len(centers)), key=lambda i: pixel_dist(centers[i], pixel))[0]


def colour_center(img, centers, colors, defaultColor=None):
    img = img.copy()
    (w, h) = img.size
    for x in range(w):
        for y in range(h):
            p = img.getpixel((x, y))
            center = closest(centers, p)
            color = colors[center]
            if defaultColor and color is None:
                color = defaultColor
            if color:
                img.putpixel((x, y), ImageColor.getrgb(color))
    return img


def fullness(img):
    (w, h) = img.size
    blacks = 0
    for x in range(w):
        for y in range(h):
            if img.getpixel((x, y))[0] < 128:
                blacks += 1
    return blacks / (w * h)


def find_laplace(img):
    ar = np.asarray(img)
    ar = cv2.cvtColor(ar, cv2.COLOR_RGB2GRAY)
    ar = cv2.Laplacian(ar, cv2.CV_64F)
    low = 0
    high = 0
    (w, h) = ar.shape
    for x in range(w):
        for y in range(h):
            bounded = min(max(ar[x, y], -10), 10)
            low = min([low, bounded])
            high = max([high, bounded])
    dist = high - low
    for x in range(w):
        for y in range(h):
            bounded = min(max(ar[x, y], -10), 10)
            ar[x, y] = math.floor((bounded - low) / dist * 255)
    return Image.fromarray(ar)


def find_sobel(img):
    img = img.copy()
    ar = np.asarray(img)
    ar = cv2.cvtColor(ar, cv2.COLOR_RGB2GRAY)
    gX = cv2.Sobel(ar, cv2.CV_64F, 1, 0, ksize=5)
    gY = cv2.Sobel(ar, cv2.CV_64F, 0, 1, ksize=5)
    magnitude = np.sqrt((gX ** 2) + (gY ** 2))
    magnitude = exposure.rescale_intensity(magnitude, in_range='image', out_range=(0, 255)).astype(np.uint8)
    return Image.fromarray(magnitude)


def apply_blur(img, blur_size=5):
    ar = np.asarray(img)
    median = cv2.medianBlur(ar, blur_size)
    return Image.fromarray(median)


def cut_out(img, w_portion, h_portion):
    (w, h) = img.size
    w_margin = (1 - w_portion) / 2
    h_margin = (1 - h_portion) / 2
    return img.crop((w_margin * w, h_margin * h, w - w_margin * w, h - h_margin * h))


def denoise(img):
    ar = np.asarray(img)
    denoised = cv2.fastNlMeansDenoisingColored(ar, None, 20, 20, 3, 10)
    return Image.fromarray(denoised)
