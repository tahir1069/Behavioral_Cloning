#Importing Some useful libraires
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2
import sklearn
def show_imgs(source_img,title1, source_img1,title2, source_img2,title3):
    """
    This wll display 3 Images
    """
    plt.figure(figsize=(6, 2))
    plt.subplot(1, 3, 1)
    plt.imshow(source_img)
    plt.title(title1)
    plt.subplot(1, 3, 2)
    plt.imshow(source_img1)
    plt.title(title2)
    plt.subplot(1, 3, 3)
    plt.imshow(source_img2)
    plt.title(title3)
    plt.show()
    print(source_img.shape)
    print(source_img1.shape)
    print(source_img2.shape)
def Load_Driving_Log(path):
    """
    This will load log data which is later on used to read images
    """    
    dataset = pd.read_csv(path+'/driving_log.csv')
    dataset.columns = ['center', 'left', 'right', 'steering', 'throttle', 'break', 'speed']
    return dataset
def Load_Images(dataset):
    """
    This will load all images as in the log/csv file recorded by the simulator 
    """
    dataset.head()
    X_train = dataset[['left', 'center', 'right']]
    X_train.columns = ['left_camera_image',
                       'center_camera_image',
                       'right_camera_image']
    Y_train = dataset['steering']
    return (X_train, Y_train) 
def SteeringAngleAdjustment(Y_train, adjustment_factor=0.2):
    """
    In original datset Labels are for center images not for right and left images
    this function will adjuust the values by adjustment factor
    """
    measurements_center = Y_train
    measurements_right = pd.Series([x + adjustment_factor for x in Y_train])
    measurements_left = pd.Series([x - adjustment_factor for x in Y_train])
    new_y_train = pd.concat([measurements_left,
                             measurements_center,
                             measurements_right],
                            axis=1)
    new_y_train.columns = ['left_camera_steering',
                           'center_camera_steering',
                           'right_camera_steering']
    return new_y_train
def JustFlattening(X_train, Y_train):
    """
    This function will flatten all of the images in a single column to 
    feed to the neural network
    """
    test = list(X_train.iloc[:, 0].values)
    test.extend(X_train.iloc[:, 1].values)
    test.extend(X_train.iloc[:, 2].values)
    
    labels = list(Y_train.iloc[:, 0].values)
    labels.extend(Y_train.iloc[:, 1].values)
    labels.extend(Y_train.iloc[:, 2].values)
    new_test = []
    for i in range(len(test)):
        test1 = test[i].rpartition('\\')
        test1 = test1[-1]
        test1 = './Data/IMG/' + test1
        new_test.append(test1)

    return new_test, labels
def generator(samples, batch_size=32):
    """
    Generate the required images and measurments for training/
    `samples` is a list of pairs (`imagePath`, `measurement`).
    """
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]
            
            images = []
            angles = []
            for imagePath, measurement in batch_samples:
                #imagePath = 'Data\\' + imagePath.strip()
                originalImage= cv2.imread(imagePath)
                image = cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB)
                images.append(image)
                angles.append(measurement)
                # Flipping
                images.append(cv2.flip(image, 1))
                angles.append(measurement * -1.0)
                
            # trim image to only see section with road
            inputs = np.array(images)
            outputs = np.array(angles)
            yield sklearn.utils.shuffle(inputs, outputs)
