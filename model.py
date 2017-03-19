import csv, cv2, os
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Cropping2D, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D



# parameters
data_path = '../P3_data/recording2/'
crop_top = 70;
crop_btm = 135;

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
                name = data_path + 'IMG/' + batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)

                # cropping
                center_image = center_image[crop_top:crop_btm,:,:]

                center_angle = float(batch_sample[3])
                imgs.append(center_image)
                angs.append(center_angle)

                # data augmentation 
                # flip images left and right
                imgs.append(np.fliplr(center_image))
                angs.append(-center_angle)

            # trim image to only see section with road
            X_train = np.array(imgs)
            y_train = np.array(angs)
            yield shuffle(X_train, y_train)


# compile and train the model using the generator function
train_generator      = generator(train_samples,      batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

ch, row, col = 3, crop_btm-crop_top, 320  # Trimmed image format 

# set up the nvidia network here
model = Sequential()
# normalization and mean centering
model.add(Lambda(lambda x:x/255.0 - 0.5, \
            input_shape  = (row, col, ch), \
            output_shape = (row, col, ch)))

# cropping applied in generator
# model.add(Cropping2D(cropping=((70, 25), (0, 0))))

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

# comppile model
model.compile(loss='mse', optimizer='adam')

# fit model
model.fit_generator(train_generator, samples_per_epoch = len(train_samples),    \
                                     validation_data   = validation_generator,  \
                                     nb_val_samples    = len(validation_samples), 
                                     nb_epoch          = 10)
# save model
model.save('model.h5')






