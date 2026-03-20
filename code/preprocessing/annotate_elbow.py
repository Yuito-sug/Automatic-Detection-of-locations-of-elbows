"""
File Name: annotate_elbow.py
  Purpose: Prepare data for a training(Annotation). Record the location of elbows by manual.
   Author: Yuito Sugimoto
"""
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np

# Load a image
img = image.load_img("/images/train/ximages/pk32.jpg",target_size=(512,512))

# The coordinate of the estimated location of the elbow.
x=65
y=350

# Mark the place of around elbow
chgd_data = np.full(1,255).reshape(1,1,1)
img = image.img_to_array(img)
img[y-5:y+5,x-5:x+5]=chgd_data

# Show to see the image
plt.imshow(image.array_to_img(img))
plt.show()
