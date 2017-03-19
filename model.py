import csv
import cv2
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Cropping2D, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

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

    measurement = float(line[3])
    measurements.append(measurement)

# augment data
images_aug, measurements_aug = [], []
# flip images left right and reverse steering angle
for image, measurement in zip(images, measurements):
    images_aug.append(image)
    images_aug.append(np.fliplr(image))

    measurements_aug.append(measurement)
    measurements_aug.append(-measurement)

print('Total samples: {}' .format(len(measurements_aug)))

X_train = np.array(images_aug)
y_train = np.array(measurements_aug)

print(X_train.shape)
print(y_train.shape)    

model = Sequential()
# normalization and mean centering
model.add(Lambda(lambda x:x/255.0-0.5, input_shape = (160,320,3)))
# -> normalized input planes 3@160x320

# cropping
model.add(Cropping2D(cropping=((70, 25), (0, 0))))

# convolutional layers
# conv 1 
model.add(Convolution2D(24, kernel5, kernel5, activation="relu"))
#model.add(MaxPooling2D())
# conv 2
model.add(Convolution2D(36, kernel5, kernel5, activation="relu"))
#model.add(MaxPooling2D())
# conv 3
model.add(Convolution2D(48, kernel5, kernel5, activation="relu"))
# conv 4
model.add(Convolution2D(64, kernel5, kernel5, activation="relu"))
# conv 5
model.add(Convolution2D(64, kernel5, kernel5, activation="relu"))

# fully connected layers
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')



model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch = 5)

model.save('model.h5')






