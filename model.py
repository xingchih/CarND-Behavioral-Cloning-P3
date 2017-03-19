import csv
import cv2
import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Lambda

data_path = '../P3_data/recording2/'

# hyper parameter
kernel5 = 5
kernel3 = 3

lines = []
csvpath = data_path + 'driving_log.csv'
with open(csvpath) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = data_path + 'IMG/' + filename
    image = cv2.imread(current_path)
    images.append(image)

    measurement = line[4]
    measurements.append(measurement)

print('Total samples: {}' .format(len(measurements)))
print(type(image))
print(type(images))
X_train = np.array(images)
y_train = np.array(measurements)

print(X_train.shape)
print(y_train.shape)    

model = Sequential()
# normalization and mean centering
model.add(Lambda(lambda x:x/255.0-0.5, input_shape = (160,320,3)))
# -> normalized input planes 3@160x320
model.add(Convolution2D(6, kernel5, kernel5, activation="relu"))
model.add(MaxPooling2D())

model.add(Convolution2D(6, kernel5, kernel5, activation="relu"))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')



model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch = 5)

model.save('model.h5')






