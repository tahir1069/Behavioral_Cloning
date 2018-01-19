#Import Some Usefull Libraries 
from helper_functions import show_imgs, Load_Driving_Log, Load_Images 
from helper_functions import SteeringAngleAdjustment, JustFlattening, generator
from models_design import getModel
import matplotlib.image as mpimg
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#Path to the dataset this whould be the same as in the cde files 
data_path = './data'
#Loading the dataset 
X_train, Y_train = Load_Images(Load_Driving_Log(data_path))
test = X_train.iloc[5].values
left_camera_image  = mpimg.imread('./data/'+test[0].strip())
center_camera_image = mpimg.imread('./data/'+test[1].strip())
right_camera_image = mpimg.imread('./data/'+test[2].strip())
#Visualizing the dataset
show_imgs(left_camera_image,'Left Camera Image',
          center_camera_image,'Center Camera Image',
          right_camera_image,'Right Camera Image')
#flipping the dataset Horizontal Flip is not used 
#this is just for visualization purpose
show_imgs(cv2.flip(center_camera_image,1),'Horizontaly Flipped Image',
          center_camera_image,'Original Image',
          cv2.flip(center_camera_image,0),'Verically Flipped Image')
#adjusting steering angles for left and right images as it is zero in original dataset
Y_train = SteeringAngleAdjustment(Y_train)
X_train, Y_train = JustFlattening(X_train, Y_train)
print('Total Images: {}'.format(len(X_train)))
print('Total Labels: {}'.format(len(Y_train)))
samples = list(zip(X_train, Y_train))
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
print('Train samples: {}'.format(len(train_samples)))
print('Validation samples: {}'.format(len(validation_samples)))
#Using Genrators to load the images it will save the memory 
#and prevent the system to be exhausted
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)
#Here nVidia. CommaAi or Mobile Net can be used
model = getModel(model="MobileNet")
# Train the model
history_object = model.fit_generator(train_generator, samples_per_epoch= \
                                     len(train_samples), 
                                     validation_data=validation_generator, \
                                     nb_val_samples=len(validation_samples),
                                     nb_epoch=2, verbose=1)
#saving the model 
model.save('model.h5')
#Checking out some errors on training and validation set
print(history_object.history.keys())
print('Loss')
print(history_object.history['loss'])
print('Validation Loss')
print(history_object.history['val_loss'])
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()  