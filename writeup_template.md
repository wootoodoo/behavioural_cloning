# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/cropped.png "cropped image"
[image2]: ./examples/data_set.png "data visualisation"
[image3]: ./examples/model_visualisation.png.png "model visualisation"
[image4]: ./examples/flipped.png "Flipped image"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I used an Neural network architecture very similar to that used by the nVidia team, after first experimenting with a single linear layer to get the simulation pipeline working, and then a Lenet architecture, before finally settling on the architecture similar to that used by the nVidia team at https://devblogs.nvidia.com/deep-learning-self-driving-cars/.

The model includes 5 RELU layers to introduce nonlinearity (code line 47, 52, 57, 62, 67), and the data is normalized in the model using a Keras lambda layer (code line 43). 

#### 2. Attempts to reduce overfitting in the model

The model contains many dropout layers, at every Convolutional layer after the RELU layer, in order to reduce overfitting (model.py lines 49, 54, 59, 64, 69). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 77 - 80). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 79).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, and also the sample training data set provided by udacity, as that set was significantly large (24108 images) and also had a steering measurement average of 0.00407, which is roughly around 0. 

The training data provided was also in BGR, and in order to train the model the data had to be 

![data visualisation][image2]
This graph shows a visualisation of the sample data set provided by udacity.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to look at similar literature for neural network and adapting it for my use case.

My first step was to use a convolution neural network model similar to the nVidia team when they implemented end-to-end deep learning training for driving an AV. I thought this model might be appropriate due to the similarity of the function of the neural network. After getting the dimensions of the training layers correct (as the height x width of the images for this exercise were different from the square images used by nVidia), the model started to perform reasonably well.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that there would be more dropout layers, and also added more training data to give the model more varied forms of driving to learn from.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, especially at the first bend close to the bridge and also at the sharp turn after crossing the bridge. To improve the driving behavior in these cases, I collected more specific data at these locations. More information on the training process can be found below.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

Here is a visualization of the architecture.

![model visualisation][image3]

#### 3. Creation of the Training Set & Training Process

To increase the amount of data collected, I used images from the centre, left and right cameras with a correction factor of 0.2 as recommended in the lesson. I experimented with smaller and larger correction factors, however it turned out that 0.2 was the most accurate for me. 

To augment the data sat, I cropped the image to prevent the details from the sky and the car bonnet to be included in the training. I also flipped images and angles thinking that this would even out the data set for turning left and right. For example, here is an image that has then been cropped and then flipped:

![cropped image][image1]
![flipped image][image4]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to correct itself when it encountered such a scenario. 

Run1 is the video of my vehicle doing one lap for track 1, with the corresponding model.h5. Run2 shows the vehicle doing one lap for track 2, with the corresponding model2.h5. As this track had plenty of sharp turns and dark corners, it was very difficult to keep the vehicle completely on the track without manual intervention. Manual intervention was required for sharp turns and areas with large shadows primarily.

Here are my training logs:
29 Nov: Training on a subset of the sample data using the nVidia framework, cleaning the data if there are empty fields in the CSV file as sometimes, the full sample data set could not be uploaded from my computer. Tuned steering correction to 0.1 and 0.3, but driving got worse. Tried removing the side cameras and driving got worse. Tried correction of 0.15, also was no good. 

1 Dec: Found the large training dataset inside the workspace and started training it with correction factor 0.2, it took about 1 hour. Started to vary the batchsize to 128, and also a learning rate of 0.001 with a decay rate of 1e-6. Learnt that I can train within the simulator workspace too. Realised that I wasn't implementing a BGR to RGB color inversion, thus explaining why the model was performing poorly. The car was able to keep to the track now completely after training on IMG3 set. Does not do well after training on IMG4 for 20 Epochs. Train for 10 epochs on IMG1 again, It is good now so save it at model.h5 (Finished). 

2 Dec: For the mountain track, I had to train a new model (model2.h5) as I realised that the terrain and road features were completely different, so it wouldn't be possible to use the same model on both tracks. Training for 20 epochs on IMG6 and save as Model2.h5 (training loss = 0.0703), training 5 epoch on IMG5 (training loss = 0.1148 but bad), training 20 epochs on IMG7 (training loss = 0.0606). Train for 5 epochs on IMG9, 20 Epochs on IMG8 and the vehicle can almost complete the full loop. Train for 5 epochs at the location on IMG 10 and IMG11, 3 times on IMG12 where the vehicle went off. Mostly due to sharp turns and bends.

4 Dec: Trained for 10 Epochs on IMG6, IMG7, IMG8, IMG13 as this image set produced the best results. The model still required a few manual interventions, especially at the sharp turns and dark corners to keep the vehicle on the track.

Track 1
IMG1 is the sample data set. (Training on 8 epochs is necessary)
IMG2 is a set of good driving on lake lap (Not used)
IMG3 is a lap of defensive driving to avoid the yellow road markings. (Used)
IMG4 is a complete lap of the track with good driving. (Not used)

Track 2
IMG5 is a set of the mountain track. (Not used)
IMG6 is more mountain track (16608 samples)
IMG7 is even more mountain track (19824 samples)
IMG8 is a good 2 laps of mountain driving. (20448 samples)
IMG9 is specific to where the model failed
IMG10, 11 and 12 is specific to where the model failed.
IMG13 is good 2 laps of mountain driving (17731 samples)

For all the different models, I randomly shuffled the data set and put 20% of the data into a validation set. I used this training data for training the model. The validation set helped determine if the model was over or under fitting. 
