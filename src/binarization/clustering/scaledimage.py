import math

from PIL import Image


class ScaledImage:
    def __init__(self, img, max_size):
        w, h = img.size
        # if image size less than max size, do nothing
        if w <= max_size and h <= max_size:
            self.scale = 1
        else:
            # resize to fit image to max size requirements
            self.scale = max_size / max(w, h)
            self.img = img.resize((math.floor(w * self.scale), math.floor(h * self.scale)), Image.ANTIALIAS)

    # given a coord, scale to fit to the scaled image
    def translate_to_scaled(self, coord):
        return coord * self.scale