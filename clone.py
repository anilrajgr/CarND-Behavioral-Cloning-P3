import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt

lines = []
with open("data/driving_log.csv") as csvfile:
  reader = csv.reader(csvfile)
  for line in reader:
    lines.append(line)
    
images = []
measurements = []
for line in lines:
  source_path = line[0]
  filename = source_path.split('\\')[-1]
  current_path = 'data/' + filename
  image = cv2.imread(current_path)
  measurement = float(line[3])
  # plt.imshow(image)
  # plt.show()
  # exit
  images.append(image)
  measurements.append(measurement)
  # Mirror image
  images.append(cv2.flip(image,1))
  measurements.append(measurement*(-1.0))

X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: x/255.0-0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,0),(0,0))))
model.add(Convolution2D(25,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,subsample=(2,2),activation="relu"))
model.add(Flatten())
model.add(Dense(1000))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
# model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=3)

model.save('model.h5')
