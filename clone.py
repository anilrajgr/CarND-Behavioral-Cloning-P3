import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt

images = []
measurements = []

##-----------------------------------------
##- The following are the images from the sample training
##  data provided at: https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip
##  Chapter 2 (Project Resources) refers to this dataset.
##-----------------------------------------
lines = []
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
## After using the above training set and various other data,
#  I found that my car is straying off right side one some curves.
#  So, I decided to get more training data at those curves
#  I decides to add (-0.5) to the angle to force it to turn left.
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

X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D

## The following is most the NVIDIA archtecture.
## I have added Dropout of 0.25 at every stage.
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
