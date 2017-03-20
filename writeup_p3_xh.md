#**Behavioral Cloning** 

##Xingchi He

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[keras_model]: ./figures/model_keras.png "Keras Model Visualization"
[nvidia_cnn]: ./figures/nvidia_cnn_architecture.png "NVidia CNN Architecture"
[img_c]: ./figures/img_c.jpg "Image from Center Camera"
[img_c_flip]: ./figures/img_c_flip.jpg "Flipped Image from Center Camera"
[img_c_crop]: ./figures/img_c_crop.jpg "Cropped Image from Center Camera"
[img_l]: ./figures/img_l.jpg "Flipped Image from Center Camera"
[img_r]: ./figures/img_r.jpg "Flipped Image from Center Camera"
[mse_5epoch]: ./figures/model_mse_loss_5epochs.png "Model MSE Loss from a 5-Epoch Training"
[mse_20epoch]: ./figures/model_mse_loss_20epochs.png "Model MSE Loss from a 20-Epoch Training"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* run1.mp4 is a video showing the car driving for two full laps in autonomous mode
* writeup_p3_xh.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

I have used the NVidia CNN architecture published [here](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/). It consists of 5 convolutional layers and 4 fully connected layers (model.py lines 86-118). The architecture is summarized in the table below

The model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.py lines 90-115) 

The model includes RELU layers to introduce nonlinearity (after each convolutional layer), and the data is normalized in the model using a Keras lambda layer (code line 90). 

####2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 30). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 118).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with a simple model and experiment with more powerful networks.

My first step was to use a simple fully connected single layer network. This model was just used to get the workflow established and tested. Then I experimented with LeNet and got the car kinda of driving for a short distance. In the end, I decided to try the NVidia CNN architechture, which has been demonstrated as a practical solution for estimating steering angle.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. The first simple model and LeNet had fairly large mean squared errors and did not perform well in the simulator. The NVidia model has shown promising results but had driven off track. I then augmented my data by adding left and right camera images and adding corrected steering angles for these images. I also incorperated the flipped images from all three cameras and their steering angles(inverted). By doing this, I have 6x data and provided the information to train the network to center the car in the lane. I decided to choose 0.5 as the correction factor. This results a small jittering in the steering angle, but enabled the car to stay robustly on the track (I had left it running for hours and it never fell off the track, just kept running and running!)

####2. Final Model Architecture

The final model architecture (model.py lines 86-118) consisted of a convolution neural network with the following layers and layer sizes 

| Layer                 |     Description                                            | 
|-----------------------|------------------------------------------------------------| 
| Input                 | 160x320x3 RGB image                                        | 
| Lambda                | outputs 160x320x3 normalized to [-1 1] and centered at 0   | 
| Cropping2D            | outputs  65x320x3                                          | 
| Convolution 5x5       | 2x2 stride, valid padding, outputs 31x158x24               |
| RELU                  |                                                            |
| Convolution 5x5       | 2x2 stride, valid padding, outputs 14x77x36                |
| RELU                  |                                                            |
| Convolution 5x5       | 2x2 stride, valid padding, outputs  5x37x48                |
| RELU                  |                                                            |
| Convolution 3x3       | 1x1 stride, valid padding, outputs  5x37x64                |
| RELU                  |                                                            |
| Convolution 3x3       | 1x1 stride, valid padding, outputs  5x37x64                |
| RELU                  |                                                            |
| Flatten               |                                                            |
| Fully connected       | Outputs 100                                                |  
| Fully connected       | Outputs 50                                                 |
| Fully connected       | Outputs 10                                                 |
| Fully connected       | Outputs 1                                                  |

Here is a visualization of the architecture, automatically generated by plot function from keras.utils.visualize_util

![alt text][keras_model]


The figure below is from [NVidia blog post](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) that provides a better visualization of the model. The annotation on the right of the figure does not reflect the size of each layer of my model. Please refer to the table above for the input/output sizes of each layer.
![alt text][nvidia_cnn]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded three laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][img_c]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would account for bias of the steering angle in clock-wise or couterclock wise driving. For example, here is an image that has then been flipped:

![alt text][img_c_flip]

After the collection process, I had 3593 number of data points. The center, left, and right camera images and their flipped versions are all used, thus I have 3593x6 samples. Below are the images from the left and right cameras, respectively.

![alt text][img_l] ![alt text][img_r]

I then preprocessed this data by normalizing to [-1, 1] centered at 0 and also applied cropping the remove the hood of the car at the bottom and the sky and the background on the top, see picure below.

![alt text][img_c_crop]

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was around 5. The two figures below show the mean squared error losses of training and validation sets evolved along epochs. The two curves intersect between first and second epochs. After 4 or 5 epochs, the MSE loss of the validation set is plateaued. Although that of the training set keeps declining, it probably indicated overfitting.

![alt text][mse_5epoch]
![alt text][mse_20epoch]

I used an adam optimizer so that manually training the learning rate wasn't necessary.
