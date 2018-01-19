
***
# Behavioural Cloning (End to End Learning) for Self Driving Cars  
---
## Overview 
---
***
Everyone in the world in alot of aspexts. Coming to driving a car here too many of us likes to drive very smoothly but some of us love driving dangerously ;). This project is all about copying one's driving style.

---
***
## Task 
---
*** 
The specific task here in this project is to predict steering angles to driving the car but we can also use throotle and break to perfectly clone someone in terms of driving.

**Input** 

•	Images and Steering Angles

**Output** 

•	Steering Angles

---
***
## Data Collection 
---
*** 

The [simulator](https://github.com/udacity/self-driving-car-sim) works in two ways:
•	Training Mode 
•	Testing Mode
In trainign mode one can drive the car around and record the data for traiing purpose. Whereas Autonomous Mode is testing mode in which one can run the trained model and see the car moving around.

---
***
## Approach 
---
*** 

Following were the steps to to complete the project.

•	Collecting the data from Simulator 

•	Data Augmentation 

•	Data Preprocessing 

•	Designing a model 

I have used three different models to train the model and see the results.

•	[nVidia end to end learning model](https://arxiv.org/pdf/1604.07316.pdf) 

•	[Comma AI](https://arxiv.org/abs/1608.01230) 

•	[Mobile Net](https://arxiv.org/abs/1704.04861) 

I trained all of three models. I was able to use nVidia and Comma.AI but not Mobile Net because after training this model whole night I was not able to test the model due to limited resources (My PC exhausted!). I am uploading model [here](http://bit.ly/2mWympR)(Hoping someone might test it and inform me :) ).
Personally in my opinion Comma.AI model gave promising results. It was more accurate and got trained in very short time. Here the videos for [track1](http://bit.ly/2mWympR) and [track2](http://bit.ly/2Dm5vBO).

As can be seen in above videos the model correctly predicts the steeriing angle but the car drives with only constatnt speed also it can not reverse if stuck somewhere. So here comes **the fun part!** 
What I did:

1.	Trained a model for steering angels with Comma.AI

2.	Trained another model for Throttle Values again this with Comma.AI 

Now I tested these models with new_drive.py. The video can be found [here](http://bit.ly/2FUkwMT).

Cool the video is a bit shaky but still seems OK. Here we can do two improvements.

•	Increasing Dropout or using some other regularization technique.

•	Recording some more fine data

*The main reason for the shaky video is not caring about throttle during data collection phase. It was hard for me to balance between throttle and angle and I was more of concerned about steering angles during data collection.

---
***
## [nVdia end to end Learning model](https://arxiv.org/pdf/1604.07316.pdf)
---
*** 

Layer (type)                 Output Shape              Param #

=================================================================

lambda_19 (Lambda)           (None, 160, 320, 3)       0
_________________________________________________________________
cropping2d_19 (Cropping2D)   (None, 90, 320, 3)        0
_________________________________________________________________
conv2d_351 (Conv2D)          (None, 43, 158, 24)       1824
_________________________________________________________________
conv2d_352 (Conv2D)          (None, 20, 77, 36)        21636
_________________________________________________________________
conv2d_353 (Conv2D)          (None, 8, 37, 48)         43248
_________________________________________________________________
conv2d_354 (Conv2D)          (None, 6, 35, 64)         27712
_________________________________________________________________
conv2d_355 (Conv2D)          (None, 4, 33, 64)         36928
_________________________________________________________________
flatten_9 (Flatten)          (None, 8448)              0
_________________________________________________________________
dense_35 (Dense)             (None, 100)               844900
_________________________________________________________________
dense_36 (Dense)             (None, 50)                5050
_________________________________________________________________
dense_37 (Dense)             (None, 10)                510
_________________________________________________________________
dense_38 (Dense)             (None, 1)                 11

=================================================================

Total params: 981,819

Trainable params: 981,819

Non-trainable params: 0

=================================================================

---
***
## [Comma AI end to end learning architecture](https://arxiv.org/abs/1608.01230)
---
*** 


Layer (type)                 Output Shape              Param #

=================================================================

lambda_20 (Lambda)           (None, 160, 320, 3)       0
_________________________________________________________________
cropping2d_20 (Cropping2D)   (None, 90, 320, 3)        0
_________________________________________________________________
conv2d_356 (Conv2D)          (None, 23, 80, 16)        3088
_________________________________________________________________
elu_9 (ELU)                  (None, 23, 80, 16)        0
_________________________________________________________________
conv2d_357 (Conv2D)          (None, 12, 40, 32)        12832
_________________________________________________________________
elu_10 (ELU)                 (None, 12, 40, 32)        0
_________________________________________________________________
conv2d_358 (Conv2D)          (None, 6, 20, 48)         38448
_________________________________________________________________
elu_11 (ELU)                 (None, 6, 20, 48)         0
_________________________________________________________________
conv2d_359 (Conv2D)          (None, 4, 18, 64)         27712
_________________________________________________________________
elu_12 (ELU)                 (None, 4, 18, 64)         0
_________________________________________________________________
conv2d_360 (Conv2D)          (None, 2, 16, 64)         36928
_________________________________________________________________
elu_13 (ELU)                 (None, 2, 16, 64)         0
_________________________________________________________________
flatten_10 (Flatten)         (None, 2048)              0
_________________________________________________________________
dense_39 (Dense)             (None, 100)               204900
_________________________________________________________________
elu_14 (ELU)                 (None, 100)               0
_________________________________________________________________
dense_40 (Dense)             (None, 50)                5050
_________________________________________________________________
elu_15 (ELU)                 (None, 50)                0
_________________________________________________________________
dense_41 (Dense)             (None, 10)                510
_________________________________________________________________
elu_16 (ELU)                 (None, 10)                0
_________________________________________________________________
dense_42 (Dense)             (None, 1)                 11

=================================================================

Total params: 329,479

Trainable params: 329,479

Non-trainable params: 0

=================================================================
 
---
***
## [Google's Mobile Net Architecture](https://arxiv.org/abs/1704.04861)
---
*** 

_________________________________________________________________
Layer (type)                 Output Shape              Param #

=================================================================

lambda_21 (Lambda)           (None, 160, 320, 3)       0
_________________________________________________________________
cropping2d_21 (Cropping2D)   (None, 90, 320, 3)        0
_________________________________________________________________
conv2d_361 (Conv2D)          (None, 44, 159, 32)       896
_________________________________________________________________
batch_normalization_325 (Bat (None, 44, 159, 32)       128
_________________________________________________________________
activation_328 (Activation)  (None, 44, 159, 32)       0
_________________________________________________________________
conv2d_362 (Conv2D)          (None, 44, 159, 32)       9248
_________________________________________________________________
batch_normalization_326 (Bat (None, 44, 159, 32)       128
_________________________________________________________________
activation_329 (Activation)  (None, 44, 159, 32)       0
_________________________________________________________________
conv2d_363 (Conv2D)          (None, 44, 159, 64)       2112
_________________________________________________________________
batch_normalization_327 (Bat (None, 44, 159, 64)       256
_________________________________________________________________
activation_330 (Activation)  (None, 44, 159, 64)       0
_________________________________________________________________
conv2d_364 (Conv2D)          (None, 21, 79, 64)        36928
_________________________________________________________________
batch_normalization_328 (Bat (None, 21, 79, 64)        256
_________________________________________________________________
activation_331 (Activation)  (None, 21, 79, 64)        0
_________________________________________________________________
conv2d_365 (Conv2D)          (None, 21, 79, 128)       8320
_________________________________________________________________
batch_normalization_329 (Bat (None, 21, 79, 128)       512
_________________________________________________________________
activation_332 (Activation)  (None, 21, 79, 128)       0
_________________________________________________________________
conv2d_366 (Conv2D)          (None, 21, 79, 128)       147584
_________________________________________________________________
batch_normalization_330 (Bat (None, 21, 79, 128)       512
_________________________________________________________________
activation_333 (Activation)  (None, 21, 79, 128)       0
_________________________________________________________________
conv2d_367 (Conv2D)          (None, 21, 79, 128)       16512
_________________________________________________________________
batch_normalization_331 (Bat (None, 21, 79, 128)       512
_________________________________________________________________
activation_334 (Activation)  (None, 21, 79, 128)       0
_________________________________________________________________
conv2d_368 (Conv2D)          (None, 11, 40, 128)       147584
_________________________________________________________________
batch_normalization_332 (Bat (None, 11, 40, 128)       512
_________________________________________________________________
activation_335 (Activation)  (None, 11, 40, 128)       0
_________________________________________________________________
conv2d_369 (Conv2D)          (None, 11, 40, 256)       33024
_________________________________________________________________
batch_normalization_333 (Bat (None, 11, 40, 256)       1024
_________________________________________________________________
activation_336 (Activation)  (None, 11, 40, 256)       0
_________________________________________________________________
conv2d_370 (Conv2D)          (None, 11, 40, 256)       590080
_________________________________________________________________
batch_normalization_334 (Bat (None, 11, 40, 256)       1024
_________________________________________________________________
activation_337 (Activation)  (None, 11, 40, 256)       0
_________________________________________________________________
conv2d_371 (Conv2D)          (None, 11, 40, 256)       65792
_________________________________________________________________
batch_normalization_335 (Bat (None, 11, 40, 256)       1024
_________________________________________________________________
activation_338 (Activation)  (None, 11, 40, 256)       0
_________________________________________________________________
conv2d_372 (Conv2D)          (None, 5, 19, 256)        590080
_________________________________________________________________
batch_normalization_336 (Bat (None, 5, 19, 256)        1024
_________________________________________________________________
activation_339 (Activation)  (None, 5, 19, 256)        0
_________________________________________________________________
conv2d_373 (Conv2D)          (None, 5, 19, 512)        131584
_________________________________________________________________
batch_normalization_337 (Bat (None, 5, 19, 512)        2048
_________________________________________________________________
activation_340 (Activation)  (None, 5, 19, 512)        0
_________________________________________________________________
conv2d_374 (Conv2D)          (None, 5, 19, 512)        2359808
_________________________________________________________________
batch_normalization_338 (Bat (None, 5, 19, 512)        2048
_________________________________________________________________
activation_341 (Activation)  (None, 5, 19, 512)        0
_________________________________________________________________
conv2d_375 (Conv2D)          (None, 5, 19, 512)        262656
_________________________________________________________________
batch_normalization_339 (Bat (None, 5, 19, 512)        2048
_________________________________________________________________
activation_342 (Activation)  (None, 5, 19, 512)        0
_________________________________________________________________
conv2d_376 (Conv2D)          (None, 5, 19, 512)        2359808
_________________________________________________________________
batch_normalization_340 (Bat (None, 5, 19, 512)        2048
_________________________________________________________________
activation_343 (Activation)  (None, 5, 19, 512)        0
_________________________________________________________________
conv2d_377 (Conv2D)          (None, 5, 19, 512)        262656
_________________________________________________________________
batch_normalization_341 (Bat (None, 5, 19, 512)        2048
_________________________________________________________________
activation_344 (Activation)  (None, 5, 19, 512)        0
_________________________________________________________________
conv2d_378 (Conv2D)          (None, 5, 19, 512)        2359808
_________________________________________________________________
batch_normalization_342 (Bat (None, 5, 19, 512)        2048
_________________________________________________________________
activation_345 (Activation)  (None, 5, 19, 512)        0
_________________________________________________________________
conv2d_379 (Conv2D)          (None, 5, 19, 512)        262656
_________________________________________________________________
batch_normalization_343 (Bat (None, 5, 19, 512)        2048
_________________________________________________________________
activation_346 (Activation)  (None, 5, 19, 512)        0
_________________________________________________________________
conv2d_380 (Conv2D)          (None, 5, 19, 512)        2359808
_________________________________________________________________
batch_normalization_344 (Bat (None, 5, 19, 512)        2048
_________________________________________________________________
activation_347 (Activation)  (None, 5, 19, 512)        0
_________________________________________________________________
conv2d_381 (Conv2D)          (None, 5, 19, 512)        262656
_________________________________________________________________
batch_normalization_345 (Bat (None, 5, 19, 512)        2048
_________________________________________________________________
activation_348 (Activation)  (None, 5, 19, 512)        0
_________________________________________________________________
conv2d_382 (Conv2D)          (None, 5, 19, 512)        2359808
_________________________________________________________________
batch_normalization_346 (Bat (None, 5, 19, 512)        2048
_________________________________________________________________
activation_349 (Activation)  (None, 5, 19, 512)        0
_________________________________________________________________
conv2d_383 (Conv2D)          (None, 5, 19, 512)        262656
_________________________________________________________________
batch_normalization_347 (Bat (None, 5, 19, 512)        2048
_________________________________________________________________
activation_350 (Activation)  (None, 5, 19, 512)        0
_________________________________________________________________
conv2d_384 (Conv2D)          (None, 3, 10, 512)        2359808
_________________________________________________________________
batch_normalization_348 (Bat (None, 3, 10, 512)        2048
_________________________________________________________________
activation_351 (Activation)  (None, 3, 10, 512)        0
_________________________________________________________________
conv2d_385 (Conv2D)          (None, 1, 8, 1024)        4719616
_________________________________________________________________
batch_normalization_349 (Bat (None, 1, 8, 1024)        4096
_________________________________________________________________
activation_352 (Activation)  (None, 1, 8, 1024)        0
_________________________________________________________________
conv2d_386 (Conv2D)          (None, 1, 4, 1024)        9438208
_________________________________________________________________
batch_normalization_350 (Bat (None, 1, 4, 1024)        4096
_________________________________________________________________
activation_353 (Activation)  (None, 1, 4, 1024)        0
_________________________________________________________________
conv2d_387 (Conv2D)          (None, 1, 4, 1024)        1049600
_________________________________________________________________
batch_normalization_351 (Bat (None, 1, 4, 1024)        4096
_________________________________________________________________
activation_354 (Activation)  (None, 1, 4, 1024)        0
_________________________________________________________________
average_pooling2d_12 (Averag (None, 1, 2, 1024)        0
_________________________________________________________________
flatten_11 (Flatten)         (None, 2048)              0
_________________________________________________________________
dropout_21 (Dropout)         (None, 2048)              0
_________________________________________________________________
dense_43 (Dense)             (None, 1)                 2049

=================================================================

Total params: 32,505,121

Trainable params: 32,483,233

Non-trainable params: 21,888

=================================================================

---
***
## Dependencies
---
*** 

This project requires Python 3.5 and the following Python libraries installed:

•	Keras

•	NumPy

•	SciPy

•	TensorFlow

•	Pandas

•	OpenCV

•	Matplotlib

---
***
## How to Run the Model
---
*** 
This repository comes with trained model which you can directly test using the following command.
Just for steering angle.

•	python drive.py model.h5 folder_name*

To run both steering angel and trottle for the model

•	python new_drive.py steering_angel_model_.h5 throttle_model.h5 folder_name*

*Folder to save the output images to create a movie.

---
***
## Conclusion and Future Work
---
The model works fine need to test it some other self driving simulators and datasets. Also collecting some fine data for both throttle, breaks and steering anglewill help the model to work in realistic way.

*Note: Currently it can work with throttle and steering angle. The model can reverse too depending on the situation. 

---
***
## References
---
*** 

•	NVIDIA model: https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/

•	Udacity Self-Driving Car Simulator: https://github.com/udacity/self-driving-car-sim
