import imageio.v2 as imageio
import numpy as np
from PIL import ImageFile
from matplotlib import pyplot as plt
from numpy import shape
from sklearn.cluster import k_means

ImageFile.LOAD_TRUNCATED_IMAGES = True

# img = imageio.imread("../HomerCompTraining/figures/homer2/txt1/P.Corn.Inv.MSS.A.101.XIII.jpg")
img = imageio.imread("../../../dataset/test/testImg1.jpg")
imgArray = np.concatenate([r for r in img])
imgY, imgX, _ = shape(img)  # get shape of image, ignore rgb
r, g, b = np.transpose(imgArray)
centroids, labels, _ = k_means(imgArray, n_clusters=3)
centroids /= 256  # convert to 0-1
centroid_colors = [centroids[l] for l in labels]

to_show = len(r)

fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter(*np.transpose(centroids), c=centroids, depthshade=False)
ax = fig.add_subplot(projection='3d')
ax.scatter(r[0:to_show], g[0:to_show], b[0:to_show], c=centroid_colors[0:to_show], depthshade=False)
plt.show()
