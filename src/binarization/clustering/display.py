import math
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

from ICFHR2022 import *
from scaledimage import ScaledImage
from imageutil import *

data = ICFHR2022()

MAX_IMG_SIZE = 1000

def show_tokens_in_context(name):
	for id in data.img_ids():
		im = data.id_to_img[id]
		f = img_file(im)
		img = Image.open(dir + f)
		print(f)
		scaled = ScaledImage(img, MAX_IMG_SIZE)
		draw = ImageDraw.Draw(scaled.img)
		anns = data.img_to_anns[id]
		drawn = False
		for a in anns:
			(cl_name, cl_base, cl_foot) = data.ann_class(a)
			if cl_name == name:
				x_min = ann_x(a)
				y_min = ann_y(a)
				x_max = ann_x(a) + ann_w(a)
				y_max = ann_y(a) + ann_h(a)
				sx_min = scaled.translate_to_scaled(x_min)
				sy_min = scaled.translate_to_scaled(y_min)
				sx_max = scaled.translate_to_scaled(x_max)
				sy_max = scaled.translate_to_scaled(y_max)
				draw.rectangle([sx_min, sy_min, sx_max, sy_max], width=2, outline='green')
				drawn = True
		if drawn:
			plt.figure(1, figsize=(10,10))
			plt.clf()
			plt.imshow(scaled.img)
			plt.pause(1)
			input("X")

def show_tokens(name, base, foot):
	_, mult = plt.subplots(1, 2, figsize = (10,5))
	for id in data.img_ids():
		print(id)
		im = data.id_to_img[id]
		f = img_file(im)
		img = Image.open(dir + f)
		scaled = ScaledImage(img, MAX_IMG_SIZE)
		anns = data.img_to_anns[id]
		n_from_image = 0
		for a in anns:
			if n_from_image < 10:
				(cl_name, cl_base, cl_foot) = data.ann_class(a)
				if cl_name == name:
					x_min = ann_x(a)
					y_min = ann_y(a)
					x_max = ann_x(a) + ann_w(a)
					y_max = ann_y(a) + ann_h(a)
					sx_min = scaled.translate_to_scaled(x_min)
					sy_min = scaled.translate_to_scaled(y_min)
					sx_max = scaled.translate_to_scaled(x_max)
					sy_max = scaled.translate_to_scaled(y_max)
					part = img.crop((ann_x(a), ann_y(a), ann_x(a) + ann_w(a), ann_y(a) + ann_h(a)))
					spart = scaled.img.crop((sx_min, sy_min, sx_max, sy_max))
					plt.figure(1)
					mult[0].imshow(part)
					mult[1].imshow(spart)
					plt.pause(1)
					n_from_image += 1

# show_tokens('Α', 'bt1', 'ft1')

# show_tokens_in_context('.')
# show_tokens_in_context('Α')
# show_tokens_in_context('Β')
# show_tokens_in_context('Γ')
# show_tokens_in_context('Δ')
# show_tokens_in_context('Ε')
# show_tokens_in_context('Ζ')
# show_tokens_in_context('Η')
show_tokens_in_context('Θ')
# show_tokens_in_context('Ι')
# show_tokens_in_context('Κ')
# show_tokens_in_context('Λ')
# show_tokens_in_context('Μ')
# show_tokens_in_context('Ν')
# show_tokens_in_context('Ξ')
# show_tokens_in_context('Ο')
# show_tokens_in_context('Π')
# show_tokens_in_context('Ρ')
# show_tokens_in_context('Τ')
# show_tokens_in_context('Υ')
# show_tokens_in_context('Φ')
# show_tokens_in_context('Χ')
# show_tokens_in_context('Ψ')
# show_tokens_in_context('Ω')
# show_tokens_in_context('Ϲ')
