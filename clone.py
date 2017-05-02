"""
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# img = cv2.imread('data/IMG/center_2016_12_01_13_30_48_287.jpg')
# img = img[50:160, 0:320]
# plt.imshow(img)
# plt.show()
# plt.imshow(crop_img)
# plt.show()
# print(img.shape)
# exit()

samples = []
with open("data/driving_log.csv") as csvfile:
  reader = csv.reader(csvfile)
  for line in reader:
    samples.append(line)
    
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

print(len(train_samples))
print(len(validation_samples))

import sklearn
import random

batch_size=32

def generator(samples, batch_size=batch_size):
  num_samples = len(samples)
  while 1:
    random.shuffle(samples)
    for offset in range(0, num_samples, batch_size):
      batch_samples = samples[offset:offset+batch_size]
      images = []
      angles = []
      for batch_sample in batch_samples:
        name = 'data/'+batch_sample[0].split('\\')[-1]
        center_image = cv2.imread(name)
        # center_image = center_image[50:160, 0:320]
        center_angle = float(batch_sample[3])
        images.append(center_image)
        angles.append(center_angle)
        # Mirror image
        images.append(cv2.flip(center_image,1))
        angles.append(center_angle*(-1.0))
      X_train = np.array(images)
      y_train = np.array(angles)
      yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

ch, row, col = 3, 160, 320

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D, Conv2D
from keras.layers.convolutional import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: x/255.0-0.5, input_shape=(row,col,ch), output_shape=(row,col,ch)))
# model.add(Cropping2D(cropping=((70,0),(0,0))))
model.add(Conv2D(25,(5,5),strides=(2,2),activation="relu"))
model.add(Dropout(0.2))
model.add(Conv2D(36,(5,5),strides=(2,2),activation="relu"))
model.add(Conv2D(48,(5,5),strides=(1,1),activation="relu"))
model.add(Dropout(0.2))
model.add(Conv2D(64,(3,3),strides=(1,1),activation="relu"))
model.add(Flatten())
model.add(Dense(1000))
model.add(Dropout(0.2))
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
# model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)
# model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=3)
model.fit_generator(train_generator, steps_per_epoch=2*len(train_samples)/batch_size, validation_data=validation_generator/batch_size, validation_steps=2*len(validation_samples), epochs=10)

model.save('model.h5')

quit()
#############################################################
"""
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt

images = []
measurements = []

##-----------------------------------------
lines = []
with open("data/driving_log.csv") as csvfile:
  reader = csv.reader(csvfile)
  for line in reader:
    lines.append(line)
    
left_image_fix = 0.17
right_image_fix = -0.17
for line in lines:
  #--- Center Image --
  current_path = 'data/' + line[0]
  image = cv2.imread(current_path)
  measurement = float(line[3])
  images.append(image)
  measurements.append(measurement)
  images.append(cv2.flip(image,1))
  measurements.append(measurement*(-1.0))
  #--- Left Image --
  current_path = 'data/' + line[1]
  image = cv2.imread(current_path)
  measurement = float(line[3])+left_image_fix
  images.append(image)
  measurements.append(measurement)
  images.append(cv2.flip(image,1))
  measurements.append(measurement*(-1.0))
  #--- Right Image --
  current_path = 'data/' + line[2]
  image = cv2.imread(current_path)
  measurement = float(line[3])+right_image_fix
  images.append(image)
  measurements.append(measurement)
  images.append(cv2.flip(image,1))
  measurements.append(measurement*(-1.0))
##-----------------------------------------
"""
lines = []
with open("data1/driving_log.csv") as csvfile:
  reader = csv.reader(csvfile)
  for line in reader:
    lines.append(line)
    
left_image_fix = 0.17
right_image_fix = -0.17
for line in lines:
  #--- Center Image --
  current_path = 'data/' + line[0]
  image = cv2.imread(current_path)
  measurement = float(line[3])
  images.append(image)
  measurements.append(measurement)
  images.append(cv2.flip(image,1))
  measurements.append(measurement*(-1.0))
  #--- Left Image --
  current_path = 'data/' + line[1]
  image = cv2.imread(current_path)
  measurement = float(line[3])+left_image_fix
  images.append(image)
  measurements.append(measurement)
  images.append(cv2.flip(image,1))
  measurements.append(measurement*(-1.0))
  #--- Right Image --
  current_path = 'data/' + line[2]
  image = cv2.imread(current_path)
  measurement = float(line[3])+right_image_fix
  images.append(image)
  measurements.append(measurement)
  images.append(cv2.flip(image,1))
  measurements.append(measurement*(-1.0))
##-----------------------------------------
lines = []
with open("data2/driving_log.csv") as csvfile:
  reader = csv.reader(csvfile)
  for line in reader:
    lines.append(line)
    
left_image_fix = 0.17
right_image_fix = -0.17
for line in lines:
  #--- Center Image --
  current_path = 'data/' + line[0]
  image = cv2.imread(current_path)
  measurement = float(line[3])
  images.append(image)
  measurements.append(measurement)
  images.append(cv2.flip(image,1))
  measurements.append(measurement*(-1.0))
  #--- Left Image --
  current_path = 'data/' + line[1]
  image = cv2.imread(current_path)
  measurement = float(line[3])+left_image_fix
  images.append(image)
  measurements.append(measurement)
  images.append(cv2.flip(image,1))
  measurements.append(measurement*(-1.0))
  #--- Right Image --
  current_path = 'data/' + line[2]
  image = cv2.imread(current_path)
  measurement = float(line[3])+right_image_fix
  images.append(image)
  measurements.append(measurement)
  images.append(cv2.flip(image,1))
  measurements.append(measurement*(-1.0))
##-----------------------------------------
lines = []
with open("data3/driving_log.csv") as csvfile:
  reader = csv.reader(csvfile)
  for line in reader:
    lines.append(line)
    
left_image_fix = 0.17
right_image_fix = -0.17
for line in lines:
  #--- Center Image --
  current_path = 'data/' + line[0]
  image = cv2.imread(current_path)
  measurement = float(line[3])
  images.append(image)
  measurements.append(measurement)
  images.append(cv2.flip(image,1))
  measurements.append(measurement*(-1.0))
  #--- Left Image --
  current_path = 'data/' + line[1]
  image = cv2.imread(current_path)
  measurement = float(line[3])+left_image_fix
  images.append(image)
  measurements.append(measurement)
  images.append(cv2.flip(image,1))
  measurements.append(measurement*(-1.0))
  #--- Right Image --
  current_path = 'data/' + line[2]
  image = cv2.imread(current_path)
  measurement = float(line[3])+right_image_fix
  images.append(image)
  measurements.append(measurement)
  images.append(cv2.flip(image,1))
  measurements.append(measurement*(-1.0))
"""
##-----------------------------------------
# Drove on the right edge of the road
# So, adding (-0.5) to the angle
lines = []
with open("data4/driving_log.csv") as csvfile:
  reader = csv.reader(csvfile)
  for line in reader:
    lines.append(line)
    
left_image_fix = 0.17
right_image_fix = -0.17
for line in lines:
  if float(line[3]) != 0.0:
    #--- Center Image --
    current_path = 'data/' + line[0]
    image = cv2.imread(current_path)
    measurement = float(line[3])-0.5
    images.append(image)
    measurements.append(measurement)
    images.append(cv2.flip(image,1))
    measurements.append(measurement*(-1.0))
    #--- Left Image --
    current_path = 'data/' + line[1]
    image = cv2.imread(current_path)
    measurement = float(line[3])+left_image_fix-0.5
    images.append(image)
    measurements.append(measurement)
    images.append(cv2.flip(image,1))
    measurements.append(measurement*(-1.0))
    #--- Right Image --
    current_path = 'data/' + line[2]
    image = cv2.imread(current_path)
    measurement = float(line[3])+right_image_fix-0.5
    images.append(image)
    measurements.append(measurement)
    images.append(cv2.flip(image,1))
    measurements.append(measurement*(-1.0))
##-----------------------------------------
"""
# Drove on the right edge of the road
# Keeping only the angles != 0.
# Adding (-0.5)
lines = []
with open("data5/driving_log.csv") as csvfile:
  reader = csv.reader(csvfile)
  for line in reader:
    lines.append(line)
    
left_image_fix = 0.17
right_image_fix = -0.17
for line in lines:
  #--- Center Image --
  current_path = 'data/' + line[0]
  image = cv2.imread(current_path)
  if float(line[3]) != 0.0:
    measurement = float(line[3])-0.5
    images.append(image)
    measurements.append(measurement)
    images.append(cv2.flip(image,1))
    measurements.append(measurement*(-1.0))
    #--- Left Image --
    current_path = 'data/' + line[1]
    image = cv2.imread(current_path)
    measurement = float(line[3])+left_image_fix-0.5
    images.append(image)
    measurements.append(measurement)
    images.append(cv2.flip(image,1))
    measurements.append(measurement*(-1.0))
    #--- Right Image --
    current_path = 'data/' + line[2]
    image = cv2.imread(current_path)
    measurement = float(line[3])+right_image_fix-0.5
    images.append(image)
    measurements.append(measurement)
    images.append(cv2.flip(image,1))
    measurements.append(measurement*(-1.0))
"""

X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: x/127.5-1.0, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Dropout(0.25))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Dropout(0.25))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Dropout(0.25))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Dropout(0.25))
model.add(Convolution2D(64,3,3,subsample=(2,2),activation="relu"))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.25))
model.add(Dense(50))
model.add(Dropout(0.25))
model.add(Dense(10))
model.add(Dropout(0.25))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
# model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=3)

model.save('model.h5')
