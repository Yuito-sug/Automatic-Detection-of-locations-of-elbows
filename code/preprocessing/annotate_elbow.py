"""
File Name: annotate_elbow.py
  Purpose: Prepare data for a training(Annotation). Record the location of elbows by manual.
   Author: Yuito Sugimoto
"""
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np

# Load a image
img = image.load_img("/images/train/pk32.jpg",target_size=(512,512))

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

# Note of the location of elows.
#pk0  (480,170),(140,145)
#pk1  (360,150),(123,266)
#pk2  (430,195),(260,275)
#pk3  (370,115),(180,185)
#pk4  (365,115),(152,122)
#pk5  (295,125),(300,106)
#pk6  (425,255),(240,190)
#pk7  (350,255),(170,90)
#pk8  (430,255),(265,210)
#pk9  (435,325),(110,168)
#pk10 (410,160),(190,160)
#pk11 (200,130) ,(175,95)
#pk12 (280,90),(148,190)
#pk13 (210,200),(225,225)
#pk14 (395,205),(308,212)
#pk15 (230,145),(145,130)
#pk16 (365,190),(240,195)
#pk17 (200,185),(134,150)

#pk18 (420,210), (150,380)
#pk19 (370,450),(70,365)
#pk20 (160,280),(85,245)
#pk21 (375,390),(110,240)
#pk22 (460,310),(185,295)
#pk23 (290,180),(180,265)
#pk24 (395,330),(85,300)
#pk25 (400,340),(100,185)
#pk26 (447,190),(260,200)
#pk27 (295,192),(263,148)
#pk28 (457,160),(97,205)
#pk29 (236,143),(100,207)
#pk30 (333,147),(155,203)
#pk31 (332,188),(192,186)
#pk32 (417,380),(65,350)

#pk33 (335,215),(179,165)
#pk34 (400,280),(85,335)
#pk35 (250,135),(300,205)
#pk36 (365,145),(350,145)

#pk37 (415,300),(155,80)
#pk38 (405,90),(345,142)
#pk39 (310,260),(200,165)
#pk40 (290,75),(100,180)
#pk41 (290,230),(180,215)
#pk42 (365,155),(155,175)

#pk43 (385,275),(157,230)
#pk44 (155,175),(215,222)
#pk45 (97,256),(395,207)
#pk46 (230,107),(318,163)

#pk47 (453,192),(240,280)
#pk48 (90,230),(407,312)
#pk49 (173,370),(389,338)
#pk50 (201,184),(302,178)
#pk51(433,332),(165,176)
#pk52 (62,275),(374,342)
# ...
