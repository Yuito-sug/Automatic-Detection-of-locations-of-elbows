"""
File Name: main_train.py
  Purpose: To train the model using input and outputs images.
   Author: Yuito Sugimoto
"""

from keras.models import Model
from keras.layers import Dense,BatchNormalization,Conv2D,Dropout,MaxPooling2D,LeakyReLU,Input,Flatten,UpSampling2D,Reshape
from keras.preprocessing import image
import numpy as np
import keras
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from keras.callbacks import EarlyStopping

# Setting of EaelyStopping
early_stopping =  EarlyStopping(
                            monitor='val_loss',
                            min_delta=0.0,
                            patience=10,
)

# Put the input images to a list.
jazz=[]
for im in range(66):
  c=image.load_img('/content/drive/MyDrive/sugimoto2/sugimoto2/pk{}.jpg'.format(im),target_size=(512,512))
  ig=image.img_to_array(c)/255
  x=image.img_to_array(ig)
  jazz.append(x)


# Put the output images to a list.
jam=[]
for am in range(66):
  a=image.load_img('/content/drive/MyDrive/Model/M{}.jpg'.format(am),target_size=(512,512))
  ig=image.img_to_array(a)/255
  xa=image.img_to_array(ig)
  jam.append(xa)


# MODEL
input_layer=Input((512,512,3))
x=Conv2D(64,kernel_size=(3,3),strides=1,padding='same')(input_layer)
x=LeakyReLU()(x)
x=Conv2D(64,(3,3),1,padding='same')(x)
x=LeakyReLU()(x)
x=MaxPooling2D((2,2))(x)
x=Conv2D(64,(3,3),1,padding='same')(x)
x=LeakyReLU()(x)
x=Dropout(0.05)(x)
x=MaxPooling2D((2,2))(x)
x=Conv2D(64,(3,3),1,padding='same')(x)
x=LeakyReLU()(x)
x=Dropout(0.05)(x)
x=MaxPooling2D((2,2))(x)
x=Conv2D(64,(3,3),1,padding='same')(x)
x=LeakyReLU()(x)
x=Dropout(0.05)(x)
x=MaxPooling2D((2,2))(x)
x=Conv2D(32,(3,3),1,padding='same')(x)
x=LeakyReLU()(x)
x=MaxPooling2D((2,2))(x)
x=Conv2D(32,(3,3),1,padding='same')(x)
x=LeakyReLU()(x)
x=Dropout(0.05)(x)
x=MaxPooling2D((2,2))(x)
x=Conv2D(32,(3,3),1,padding='same')(x)
x=LeakyReLU()(x)
x=MaxPooling2D((2,2))(x)
x=Flatten()(x)

x=Dense(128)(x)
x=LeakyReLU()(x)
x=Reshape((4,4,8))(x)
x=Conv2D(32,kernel_size=(5,5),strides=1,padding='same')(x)
x=LeakyReLU()(x)
x=Dropout(0.05)(x)
x=UpSampling2D((2,2))(x)
x=Conv2D(64,kernel_size=(5,5),strides=1,padding='same')(x)
x=LeakyReLU()(x)
x=Dropout(0.05)(x)
x=UpSampling2D((2,2))(x)
x=Conv2D(64,kernel_size=(5,5),strides=1,padding='same')(x)
x=LeakyReLU()(x)
x=Dropout(0.05)(x)
x=UpSampling2D((2,2))(x)
x=Conv2D(64,kernel_size=(5,5),strides=1,padding='same')(x)
x=LeakyReLU()(x)
x=UpSampling2D((2,2))(x)
x=Conv2D(64,kernel_size=(5,5),strides=1,padding='same')(x)
x=LeakyReLU()(x)
x=Dropout(0.05)(x)
x=UpSampling2D((2,2))(x)
x=Conv2D(64,kernel_size=(5,5),strides=1,padding='same')(x)
x=LeakyReLU()(x)
x=UpSampling2D((2,2))(x)
x=Conv2D(64,kernel_size=(5,5),strides=1,padding='same')(x)
x=LeakyReLU()(x)
x=Dropout(0.05)(x)
x=UpSampling2D((2,2))(x)
x=Conv2D(3,kernel_size=(5,5),strides=1,padding='same')(x)
x=LeakyReLU()(x)

model=Model(input_layer,x)


# Convert a Python standard list to a NumPy's list.
jazz=np.array(jazz)
jam=np.array(jam)

# Compile the model
model.compile(optimizer='adam',loss='binary_crossentropy')
# Train the model
model.fit(jazz,jam,batch_size=6,epochs=30,validation_split=0.2, callbacks=[early_stopping])

# Save the model.
model.save('detect_elbow.h5')
