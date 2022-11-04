import math
import random

import cv2
# Loading the image
import numpy as np
from sklearn.cluster import DBSCAN

from src.util.img_util import random_color


def get_min_points(key_points):
    return len(key_points) // 25


def get_epsilon(key_points, min_points=2):
    # Using algorithm from Daszykowski
    random_points = [(random.randint(0, w), random.randint(0, h)) for _ in key_points]
    return np.percentile([kth_nearest_neighbor_dist(p, random_points, min_points) for p in random_points], 95)


def kth_nearest_neighbor_dist(point, points, min_points=2):
    dists = sorted([math.dist(point, p) for p in points])
    # return mean of n+1 smallest dists, as 0 will be included
    return dists[min_points]


img = cv2.imread("C:\\Users\\Carson Brown\\git\\glyphs\\dataset\\train\\resized\\binarized\\901045_0007.png")
w, h, _ = img.shape
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Applying SIFT detector
sift = cv2.SIFT_create()
key_points, descriptors = sift.detectAndCompute(gray, None)
key_points = [np.array(kp.pt, int) for kp in key_points]
min_points = get_min_points(key_points)
ε = get_epsilon(key_points, min_points)
clusters = DBSCAN(eps=ε, min_samples=min_points).fit(descriptors)
print(clusters)

last_label = -2
color = random_color(in_order=True)
for i, label in enumerate(sorted(clusters.labels_)):
    if label == -1:
        continue
    if label != last_label:
        color = random_color(in_order=True)
        last_label = label
    img = cv2.circle(img, key_points[i], 1, color=color, thickness=5)

cv2.imshow('image-with-keypoints.jpg', img)
cv2.waitKey()
