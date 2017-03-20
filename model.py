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

correction = 0.5 # this is a parameter to tune
num_epoch = 25
crop_top = 70
crop_btm = 25

ch, row, col = 3, 160, 320  # image format 

csvpath = data_path + 'driving_log.csv'
samples = []
with open(csvpath) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)


# create adjusted steering measurements for the side camera images
            
            steering_left = steering_center + correction
            steering_right = steering_center - correction

            # read in images from center, left and right cameras
            directory = "..." # fill in the path to your training IMG directory
            img_center = process_image(np.asarray(Image.open(path + row[0])))
            img_left = process_image(np.asarray(Image.open(path + row[1])))
            img_right = process_image(np.asarray(Image.open(path + row[2])))

            # add images and angles to data set
            car_images.extend(img_center, img_left, img_right)
            steering_angles.extend(steering_center, steering_left, steering_right)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            imgs = [] # images
            angs = [] # steering angles
            for batch_sample in batch_samples:
                name_c = data_path + 'IMG/' + batch_sample[0].split('/')[-1]
                name_l = data_path + 'IMG/' + batch_sample[1].split('/')[-1]
                name_r = data_path + 'IMG/' + batch_sample[2].split('/')[-1]
                image_c = cv2.imread(name_c) # center image
                image_l = cv2.imread(name_l) # left image
                image_r = cv2.imread(name_r) # right image

                # cropping
                #center_image = center_image[crop_top:crop_btm,:,:]

                angle_c = float(batch_sample[3]) # center steering angle
                angle_l = angle_c + correction # left corrected steering anlge
                angle_r = angle_c - correction # right corrected steering angle

                imgs.append(image_c)
                angs.append(angle_c)
                
                imgs.append(image_l)
                angs.append(angle_l)

                imgs.append(image_r)
                angs.append(angle_r)

                # data augmentation 
                # flip images left and right
                imgs.append(np.fliplr(image_c))
                angs.append(-angle_c)

                imgs.append(np.fliplr(image_l))
                angs.append(-angle_l)

                imgs.append(np.fliplr(image_r))
                angs.append(-angle_r)

            # trim image to only see section with road
            X_train = np.array(imgs)
            y_train = np.array(angs)
            yield shuffle(X_train, y_train)


# compile and train the model using the generator function
train_generator      = generator(train_samples,      batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# set up the nvidia network here
model = Sequential()
# normalization and mean centering
#model.add(Flatten(input_shape=(row,col,ch)))
model.add(Lambda(lambda x:x/127.5 - 1., \
            input_shape  = (row, col, ch))

# cropping applied in generator
model.add(Cropping2D(cropping=((crop_top, crop_btm), (0, 0))))

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
model.fit_generator(train_generator, samples_per_epoch = 6*len(train_samples),    \
                                     validation_data   = validation_generator,  \
                                     nb_val_samples    = 6*len(validation_samples), 
                                     nb_epoch          = 10)
# save model
model.save('model.h5')






