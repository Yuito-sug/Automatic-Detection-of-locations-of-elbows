"""
File Name: test_model.py
  Purpose: Test the model to know how good the model is.
   Author: Yuito Sugimoto
"""
import keras
from keras.preprocessing import image
import numpy as np

# Load the trained model
model = keras.models.load_model('detect_elbow.h5', compile=True)

# Put the images to a list.
test_images = []
for i in range(10):
  img = image.load_img(f'test{i}.jpg',target_size=(512,512))
  img_array = image.img_to_array(img)/255
  img_array = image.img_to_array(img_array)
  test_images.append(img_array)
test_images = np.array(test_images)

# Pass the input image 
result = model.predict(test_images)
import matplotlib.pyplot as plt

# Show the result image to confirm if the model works well.
for i in range(10):
  img = image.load_img(f'test{i}.jpg',target_size=(512,512))
  plt.imshow(img)
  plt.imshow(image.array_to_img(result[i]),alpha=0.9)
  plt.show()
