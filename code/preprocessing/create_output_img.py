"""
File Name: create_output_img.py
  Purpose: Make a desired output againt a input image. Make a image that the location of elbows is marked.
   Author: Yuito Sugimoto
"""
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import cv2


heat_w, heat_h = (200, 200)
X, Y = np.mgrid[0:heat_w:200j, 0:heat_h:200j]
pos = np.dstack((X, Y))
mu = np.array([heat_w//2, heat_h//2])
sigma = np.array([[1000, 10], [10, 1000]])
lmd = 10000
Z = lmd * multivariate_normal(mu, sigma).pdf(pos)
img = plt.imread("/images/train/ximages/pk51.jpg")

# Resize a image
height = 512
width = 512
img = cv2.resize(img, (width, height))

heatmap_img = np.zeros((height, width))

# Center point of the heatmap.
cx, cy = (62,275)
bx,by=(374,342)

for i in range(heat_h):
    for j in range(heat_w):
        if(cy - heat_h//2 + i < 0) or (cy - heat_h//2 + i > height - 1):
            continue
        if(cx - heat_w//2 + j < 0) or (cx - heat_w//2 + j > width - 1):
            continue
        heatmap_img[cy - heat_h//2 + i, cx - heat_w//2 + j] = min(255, heatmap_img[cy - heat_h//2 + i, cx - heat_w//2 + j] + Z[i, j])

heatmap2_img = np.zeros((height, width))
for si in range(heat_h):
    for jn in range(heat_w):
        if(by - heat_h//2 + si < 0) or (by - heat_h//2 + si > height - 1):
            continue
        if(bx - heat_w//2 + jn < 0) or (bx - heat_w//2 + jn > width - 1):
            continue
        heatmap2_img[by - heat_h//2 + si, bx - heat_w//2 + jn] = min(255, heatmap2_img[by - heat_h//2 + si, bx - heat_w//2 + jn] + Z[si, jn])

# Show the image to confirm if the output image is correct.
plt.imshow(img)
heatmap_img=cv2.add(heatmap_img,heatmap2_img)
# Overlay 'heatmap_img' onto the image with an opacity of 0.8.
plt.imshow(heatmap_img, alpha=0.8)

# Show it.
plt.show()

# Save the output image.
from keras.preprocessing import image
heatmap_img.shape=((512,512,1))
image.save_img("/images/train/yimages/M51.jpg",heatmap_img)
