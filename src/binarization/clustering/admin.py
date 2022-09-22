import math
import re
import json
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFile, ImageColor, ImageFont
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from statistics import mean

from ICFHR2022 import *
from imageutil import *

ImageFile.LOAD_TRUNCATED_IMAGES = True
font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeSans.ttf", size=20)

def get_thresholds(page):
	(page_w, page_h) = page.size
	vals = []
	for x in range(0, page_w, 2):
		for y in range(0, page_h, 2):
			vals.append(page.getpixel((x,y)))
	kmeans = KMeans(n_clusters=3,n_init=3, max_iter=10).fit(vals)
	centers = sorted(kmeans.cluster_centers_, key=lambda t: t[0]+t[1]+t[2])
	ink_center = centers[0]
	paper_center = centers[1]
	background_center = centers[2]
	background_threshold = average_pixel_weighted(background_center, paper_center, 1)
	ink_threshold = average_pixel_weighted(background_center, ink_center, 20)
	return (ink_center, background_threshold, ink_threshold)

def scaled_image(im):
	page = Image.open(dir + img_file(im))
	(page_w, page_h) = page.size
	MAX_W = 2000
	if page_w > MAX_W:
		ratio = MAX_W / page_w
		page = page.resize((math.floor(page_w * ratio), math.floor(page_h * ratio)), Image.ANTIALIAS)
	return page

def colour_background(page):
	(page_w, page_h) = page.size
	for x in range(page_w):
		for y in range(page_h):
			pix = page.getpixel((x,y))
			if lighter_pixel(pix, background_threshold):
				page.putpixel((x,y), ImageColor.getrgb("blue"))

def colour_ink(page):
	(page_w, page_h) = page.size
	for x in range(page_w):
		for y in range(page_h):
			pix = page.getpixel((x,y))
			if pixel_dist(ink_threshold, pix) < 75:
				# if lighter_pixel(ink_threshold, pix):
				page.putpixel((x,y), ImageColor.getrgb("red"))

if False:
	for a in annotations:
		cat_id = ann_cat_id(a)
		cat = id_to_cat[cat_id]
		name = cat_name(cat)
		if len(name) != 1:
			print(name)

im = images[10]
im_id = img_id(im)
anns = img_to_anns[im_id]

page = scaled_image(im)

plt.figure(0)
plt.imshow(page)

(ink_center, background_threshold, ink_threshold) = get_thresholds(page)

colour_background(page)
colour_ink(page)

if False:
	lines = divide_into_lines(anns)
	for line in lines:
		for a in line:
			cat_id = ann_cat_id(a)
			cat = id_to_cat[cat_id]
			name = cat_name(cat)
			print(a, name, ord(name))

def add_boxes(page):
	draw = ImageDraw.Draw(page)
	for an in anns:
		x = ann_x(an)
		y = ann_y(an)
		w = ann_w(an)
		h = ann_h(an)
		cat_id = ann_cat_id(an)
		cat = id_to_cat[cat_id]
		name = cat_name(cat)
		
		draw.rectangle([x, y, x+w, y+h], width=1)
		draw.text((x,y), name, 'white', align='left', font=font)
		print(ord(name), name)
	i = 100
	j = 100

add_boxes(page)
plt.figure(1)
plt.imshow(page)
plt.pause(1000)
