# Import all bibliotheques that is necessair to the projet
import os
import csv   
import cv2   

import sklearn
import numpy as np    
from random import shuffle      
import matplotlib.pyplot as plt
             
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers import Lambda, Cropping2D
from keras.layers.convolutional import Conv2D       
from keras.layers.pooling import MaxPooling2D                        
  
##################### FUNCTIONS-DEFINITION ############################
# Normalizing data and mean centering the data (from course in lession "Data Preprossing")
# https://classroom.udacity.com/nanodegrees/nd013/parts/168c60f1-cc92-450a-a91b-e427c326e6a7/modules/6b6c37bc-13a5-47c7-88ed-eb1fce9789a0/lessons/3fc8dd70-23b3-4f49-86eb-a8707f71f8dd/concepts/26433503-3f0c-4eb4-a73c-84a8c2607a3c
def pre_processData(data):
    return (data/255.0) - 0.5

# Data augmentation is to reverse the images and direction measurements to help with left turn bias (From course in lession "Data Augmentation")
# https://classroom.udacity.com/nanodegrees/nd013/parts/168c60f1-cc92-450a-a91b-e427c326e6a7/modules/6b6c37bc-13a5-47c7-88ed-eb1fce9789a0/lessons/3fc8dd70-23b3-4f49-86eb-a8707f71f8dd/concepts/580c6a1d-9d20-4d2e-a77d-755e0ca0d4cd
def data_augmentation(images, angles):
    augmented_images, augmented_angles = [], []
    for image, angle in zip(images, angles):
        augmented_images.append(image)
        image_flipped = cv2.flip(image,1)
        augmented_angles.append(angle)     
        angle_flipped = angle * (-1)
        augmented_images.append(image_flipped)
        augmented_angles.append(angle_flipped)     
    return augmented_images, augmented_angles 
    
# The idea is to use all three camera images to train the model, for that create adjusted steering measurements for the side camera images and add images and angles to data set (From course in lessons "Using Multiple Cameras")
# https://classroom.udacity.com/nanodegrees/nd013/parts/168c60f1-cc92-450a-a91b-e427c326e6a7/modules/6b6c37bc-13a5-47c7-88ed-eb1fce9789a0/lessons/3fc8dd70-23b3-4f49-86eb-a8707f71f8dd/concepts/2cd424ad-a661-4754-8421-aec8cb018005
def multiple_camera(batch_samples):
    images = []
    angles = [] 
    for batch_sample in batch_samples:
        correction = 0.2
        #Center_image
        source_path = batch_sample[0]
        filename = source_path.split('/')[-1]
        current_path = './data_final/IMG/' + filename
        center_image = cv2.imread(current_path) 
        center_angle = float(batch_sample[3]) 
        images.append(center_image)
        angles.append(center_angle)     
        #Left_image
        source_path = batch_sample[1]
        filename = source_path.split('/')[-1]
        current_path = './data_final/IMG/' + filename
        left_image = cv2.imread(current_path) 
        images.append(left_image)
        angles.append(center_angle + correction)        
        #rigth_image
        source_path = batch_sample[2]
        filename = source_path.split('/')[-1]
        current_path = './data_final/IMG/' + filename
        right_image = cv2.imread(current_path) 
        images.append(right_image)
        angles.append(center_angle - correction)   
    return images, angles 
 
# Here we extract pieces of data and process them on the fly only with the generator function (From example givin in course "Generators")
# https://classroom.udacity.com/nanodegrees/nd013/parts/168c60f1-cc92-450a-a91b-e427c326e6a7/modules/6b6c37bc-13a5-47c7-88ed-eb1fce9789a0/lessons/3fc8dd70-23b3-4f49-86eb-a8707f71f8dd/concepts/b602658e-8a68-44e5-9f0b-dfa746a0cc1a
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]  
            # call multiple-camera function to add the tree images
            images, angles = multiple_camera(batch_samples)
            # Call augmented_images function to flip and mean centered data
            augmented_images, augmented_angles = data_augmentation(images, angles)
            # 
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_angles)
            yield sklearn.utils.shuffle(X_train, y_train) 

# This is function that represente the network architecture of Nvidia (See on the course "Even More Powerful Network")
# https://classroom.udacity.com/nanodegrees/nd013/parts/168c60f1-cc92-450a-a91b-e427c326e6a7/modules/6b6c37bc-13a5-47c7-88ed-eb1fce9789a0/lessons/3fc8dd70-23b3-4f49-86eb-a8707f71f8dd/concepts/7f68e171-cf87-40d2-adeb-61ae99fe56f5
def Model_Nvidia():
    model = Sequential()
    model.add(Lambda(lambda x: (x/255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70,25),(0,0))))  
    model.add(Conv2D(24, (5, 5), activation='relu',strides=(2, 2)))
    #model.add(MaxPooling2D())
    model.add(Conv2D(36, (5, 5), activation='relu',strides=(2, 2)))
    #model.add(MaxPooling2D())
    model.add(Conv2D(48, (5, 5), activation='relu',strides=(2, 2)))
    #model.add(MaxPooling2D())
    model.add(Conv2D(64, (3, 3), activation='relu',strides=(1, 1))) 
    model.add(Flatten())
    model.add(Dense(1164))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu')) 
    model.add(Dropout(0.5))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.5))    
    model.add(Dense(10, activation='relu'))  
    model.add(Dropout(0.5))    
    model.add(Dense(1)) 
    return model

# This is function that represente the network architecture of LeNet 
# https://classroom.udacity.com/nanodegrees/nd013/parts/168c60f1-cc92-450a-a91b-e427c326e6a7/modules/6b6c37bc-13a5-47c7-88ed-eb1fce9789a0/lessons/fe16d91d-bc00-455b-bc7b-a61b5e55d473/concepts/badd25bd-e30d-4b07-a4ee-bca38280c067
def Model_Lenet():   
    model = Sequential()
    model.add(Lambda(lambda x: (x/255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70,25),(0,0))))
    model.add(Conv2D(6, (5,5), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(6, (5,5), activation='relu'))
    model.add(MaxPooling2D()) 
    model.add(Flatten())
    model.add(Dense(120))  
    model.add(Dense(84))
    model.add(Dense(1))    
    return model
 
########################### END-DEFINITION-FUNCTION ######################################

# Here we using python csv library to read and store the lines from the "driving_log.csv" file, then for each line extract the path to the camera image (From course in lessons "Training Your Network" video: 0:20s - 0.35s)
# https://classroom.udacity.com/nanodegrees/nd013/parts/168c60f1-cc92-450a-a91b-e427c326e6a7/modules/6b6c37bc-13a5-47c7-88ed-eb1fce9789a0/lessons/3fc8dd70-23b3-4f49-86eb-a8707f71f8dd/concepts/b6356fc5-5191-40ae-a2d9-3c8d2c2b37bb
samples = [] 
with open('./data_final/driving_log.csv') as csvfile: 
    reader = csv.reader(csvfile)
    for line in reader: 
        samples.append(line) 
      
        
# Here we using sklearn bibliotheque to use 80% of your data for training and 20% for validation      
#shuffle(samples)
train_samples, validation_samples = train_test_split(samples, test_size=0.25) 

# Display of data collected, training and validation.
print('Data Collect: {}'.format(len(samples)))
print('Train samples: {}'.format(len(train_samples)))   
print('Validation samples: {}'.format(len(validation_samples)))
            
# Compile and train the model using the generator function  
train_generator = generator(train_samples)
validation_generator = generator(validation_samples)
model = Model_Nvidia()
#model = Model_Lenet()
model.compile(loss='mse',optimizer='adam')
history_object = model.fit_generator(train_generator, steps_per_epoch=len(train_samples), validation_data=validation_generator, validation_steps=len(validation_samples), epochs=5, verbose=1)

# Here we save the model 
model.save('modelNvidia.h5')
print('Model save succesful')

# keras method to print the model summary 
model.summary()  
                      
# Print the Loss and Validation Loss contained in the history object
print('Loss')  
print(history_object.history['loss'])
print('Validation Loss') 
print(history_object.history['val_loss'])   

### Plot the training and validation loss for each epoch     
#plt.plot(history_object.history['loss'])
#plt.plot(history_object.history['val_loss'])
#plt.title('model mean squared error loss')
#plt.ylabel('mean squared error loss')
#plt.xlabel('epoch')
#plt.legend(['training set', 'validation set'], loc='upper right')   
#plt.show()                      
exit()            