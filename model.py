import csv
import cv2
import numpy as np

from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Cropping2D, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D



# parameters
data_path = '../P3_data/recording2/'


csvpath = data_path + 'driving_log.csv'
samples = []
with open(csvpath) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            imgs = [] # images
            angs = [] # steering angles
            for batch_sample in batch_samples:
                name = data_path = 'IMG/' + batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                imgs.append(center_image)
                angs.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

ch, row, col = 3, 80, 320  # Trimmed image format

# hyper parameter
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
model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation="relu"))
#model.add(MaxPooling2D())
# conv 2
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation="relu"))
#model.add(MaxPooling2D())
# conv 3
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation="relu"))
# conv 4
model.add(Convolution2D(64, 3, 3, activation="relu"))
# conv 5
model.add(Convolution2D(64, 3, 3, activation="relu"))

# fully connected layers
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')



model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch = 5)

model.save('model.h5')






