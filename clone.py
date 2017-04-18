import csv
import cv2
import numpy as np

lines = []
with open("data/driving_log.csv") as csvfile:
  reader = csv.reader(csvfile)
  for line in reader:
    lines.append(line)
    
c_images = []
l_images = []
r_images = []
measurements = []
for line in lines:
  source_path = line[0]
  filename = source_path.split('\\')[-1]
  current_path = 'data/IMG/' + filename
  image = cv2.imread(current_path)
  c_images.append(image)
  ## source_path = line[1]
  ## filename = source_path.split('\\')[-1]
  ## current_path = 'data/IMG/' + filename
  ## image = cv2.imread(current_path)
  ## l_images.append(image)
  ## source_path = line[2]
  ## filename = source_path.split('\\')[-1]
  ## current_path = 'data/IMG/' + filename
  ## image = cv2.imread(current_path)
  ## r_images.append(image)
  measurement = float(line[3])
  measurements.append(measurement)

# X_train = [np.array(l_images), np.array(c_images), np.array(r_images)]
X_train = np.array(c_images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Merge
from keras.layers import Flatten, Dense, Lambda
# from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D

## left_cam = Sequential()
## left_cam.add(Lambda(lambda x: x/255.0-0.5, input_shape=(160,320,3)))
## left_cam.add(Conv2D(6,(5,5),activation="relu"))
## left_cam.add(MaxPooling2D())
## left_cam.add(Conv2D(6,(5,5),activation="relu"))
## left_cam.add(MaxPooling2D())
## left_cam.add(Flatten())
## left_cam.add(Dense(1))
## 
## right_cam = Sequential()
## right_cam.add(Lambda(lambda x: x/255.0-0.5, input_shape=(160,320,3)))
## right_cam.add(Conv2D(6,(5,5),activation="relu"))
## right_cam.add(MaxPooling2D())
## right_cam.add(Conv2D(6,(5,5),activation="relu"))
## right_cam.add(MaxPooling2D())
## right_cam.add(Flatten())
## right_cam.add(Dense(1))
## 
## center_cam = Sequential()
## center_cam.add(Lambda(lambda x: x/255.0-0.5, input_shape=(160,320,3)))
## center_cam.add(Conv2D(6,(5,5),activation="relu"))
## center_cam.add(MaxPooling2D())
## center_cam.add(Conv2D(6,(5,5),activation="relu"))
## center_cam.add(MaxPooling2D())
## center_cam.add(Flatten())
## center_cam.add(Dense(1))

model = Sequential()
model.add(Lambda(lambda x: x/255.0-0.5, input_shape=(160,320,3)))
# model.add(Convolution2D(6,5,5,activation="relu"))
model.add(Conv2D(6,(5,5),activation="relu"))
model.add(MaxPooling2D())
model.add(Conv2D(6,(5,5),activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(1))

## model = Sequential()
## model.add(Merge([left_cam, center_cam, right_cam], mode='concat'))
## model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
# model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=3)
## model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=2)

model.save('model.h5')
