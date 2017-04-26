import csv
import cv2
import numpy as np

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
  current_path = 'data/IMG/' + filename
  image = cv2.imread(current_path)
  images.append(image)
  measurement = float(line[3])
  measurements.append(measurement)
  image_flipped = np.fliplr(image)
  measurement_flipped = -measurement
  images.append(image_flipped)
  measurements.append(measurement_flipped)

X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Merge
# from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
import simplejson

 
m1 = Sequential() 
m1.add(Lambda(lambda x: x/255.0-0.5, input_shape=(160,320,3))) 
m1.add(Conv2D(20,(5,5),activation="relu")) 
m1.add(MaxPooling2D()) 
m1.add(Conv2D(60,(5,5),activation="relu")) 
m1.add(MaxPooling2D()) 
m1.add(Flatten()) 
m1.add(Dense(1)) 
 
m2 = Sequential() 
m2.add(Lambda(lambda x: x/255.0-0.5, input_shape=(160,320,3))) 
m2.add(Conv2D(60,(5,5),activation="relu")) 
m2.add(MaxPooling2D()) 
m2.add(Conv2D(20,(5,5),activation="relu")) 
m2.add(MaxPooling2D()) 
m2.add(Flatten()) 
m2.add(Dense(1)) 
 
m3 = Sequential() 
m3.add(Lambda(lambda x: x/255.0-0.5, input_shape=(160,320,3))) 
m3.add(Conv2D(100,(5,5),activation="relu")) 
m3.add(MaxPooling2D()) 
m3.add(Conv2D(100,(5,5),activation="relu")) 
m3.add(MaxPooling2D()) 
m3.add(Conv2D(100,(5,5),activation="relu")) 
m3.add(MaxPooling2D()) 
m3.add(Conv2D(100,(5,5),activation="relu")) 
m3.add(MaxPooling2D()) 
m3.add(Flatten()) 
m3.add(Dense(1)) 

model = Sequential() 
model.add(Merge([m1, m2, m3], mode='concat')) 
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
# model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)
model.fit([X_train, X_train, X_train], y_train, validation_split=0.2, shuffle=True, epochs=3)
# model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=10)

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(simplejson.dumps(simplejson.loads(model_json), indent=4))
model.save_weights('model.h5')
# model.save('model.h5')
