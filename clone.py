import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

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
        plt.imshow(center_image)
        plt.show()
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
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D, Conv2D
from keras.layers.convolutional import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: x/255.0-0.5, input_shape=(row,col,ch), output_shape=(row,col,ch)))
model.add(Cropping2D(cropping=((70,0),(0,0))))
model.add(Conv2D(25,(5,5),strides=(2,2),activation="relu"))
model.add(Conv2D(36,(5,5),strides=(2,2),activation="relu"))
model.add(Conv2D(48,(5,5),strides=(2,2),activation="relu"))
model.add(Conv2D(64,(3,3),strides=(2,2),activation="relu"))
model.add(Conv2D(64,(3,3),strides=(2,2),activation="relu"))
model.add(Flatten())
model.add(Dense(1000))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
# model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)
# model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=3)
model.fit_generator(train_generator, steps_per_epoch=2*len(train_samples)/batch_size, validation_data=validation_generator, validation_steps=2*len(validation_samples)/batch_size, epochs=3)

model.save('model.h5')
