#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 10:26:56 2021

@author: thomas
"""
import numpy as np 
np.random.seed(100) 
import cv2 
import os 
from PIL import Image 
import tensorflow as tf 
 
from tensorflow import keras 


# preprocessing the image 
image_dir = 'cell_images/' 
SIZE = 64 
dataset = [] 
label = [] 

# reading the image 
parasitized_images = os.listdir(image_dir+'Parasitized/') 
for i, image_name in enumerate(parasitized_images):
    if (image_name.split('.'))[1] == 'png':
        image = cv2.imread(image_dir+'Parasitized/'+image_name)
        image = Image.fromarray(image, 'RGB') 
        image = image.resize((SIZE, SIZE))
        dataset.append(np.array(image))
        label.append(0) 
    

uninfected_images = os.listdir(image_dir+'Uninfected/') 
for i, image_name in enumerate(uninfected_images):
    if (image_name.split('.'))[1] == 'png':
        image = cv2.imread(image_dir+'Uninfected/'+image_name)
        image = Image.fromarray(image, 'RGB') 
        image = image.resize((SIZE, SIZE)) 
        dataset.append(np.array(image))
        label.append(1)
        

# architecutre 
# input --> conv+pool+dropout ---> conv+pool+ dropout ---> flatten 
# ---> dense layer + norma+ dropout ---> another dense ----> output 

# optimiser - adam 
# loss - crossentropy 

INPUT_SHAPE = (SIZE,SIZE,3) # Change to (size, size, channels) 
inp = keras.layers.Input( shape = INPUT_SHAPE) 

# DEFINING THE NETWORK 
# first block  
conv1 = keras.layers.Conv2D(32,(3,3), 
                            activation = 'relu', 
                            padding = 'same')(inp) 
pool1 = keras.layers.MaxPooling2D(pool_size = (2,2))(conv1) 
norm1 = keras.layers.BatchNormalization(axis = -1)(pool1) 
drop1 = keras.layers.Dropout(rate = 0.2)(norm1)  # regularization 

# second block 
conv2 = keras.layers.Conv2D(32,(3,3), 
                            activation = 'relu', 
                            padding = 'same')(drop1) 
pool2 = keras.layers.MaxPooling2D(pool_size = (2,2))(conv2) 
norm2 = keras.layers.BatchNormalization(axis = -1)(pool2) 
drop2 = keras.layers.Dropout(rate = 0.2)(norm2) 

# flattening weights 
flat = keras.layers.Flatten()(drop2) 

# fully connected set 1 
hidden1 = keras.layers.Dense(512, activation = 'relu')(flat) 
norm3 = keras.layers.BatchNormalization(axis = -1)(hidden1) 
drop3 = keras.layers.Dropout(rate = 0.2)(norm3) 

# fully connected set 2 
hidden2 = keras.layers.Dense(256, activation = 'relu')(drop3) 
norm4 = keras.layers.BatchNormalization(axis= -1)(hidden2) 
drop4 = keras.layers.Dropout(rate = 0.2)(norm4) 

# output layer 
out = keras.layers.Dense(2, activation = 'sigmoid')(drop4) 


# define the model 
model = keras.Model(inputs = inp , outputs = out) 
model.compile(optimizer = 'adam', 
              loss = 'categorical_crossentropy', 
              metrics = ['accuracy'])

print(model.summary()) 



# dividing the data into training and testing 
from sklearn.model_selection import train_test_split 
from keras.utils import to_categorical 

X_train, X_test, y_train, y_test = train_test_split(
    dataset, to_categorical(np.array(label)), 
    test_size = 0.20,
    random_state = 0
    )


history = model.fit(np.array(X_train),
                    y_train, 
                    batch_size = 64, verbose = 1, epochs = 10,
                    validation_split = 0.1, shuffle = False) 

print('Test_Accuracy: {:.2f}%'.format(model.evaluate(np.array(X_test), np.array(y_test))[1]*100) ) 

import matplotlib.pyplot as plt 
f, (ax1, ax2) = plt.subplots(1,2, figsize=(12, 4 ))
t = f.suptitle('CNN PERFORMANCE', fontsize = 12) 
f.subplots_adjust(top = 0.85, wspace = 0.3) 

max_epoch = len(history.history['accuracy'])+1 
epoch_list = list(range(1, max_epoch)) 
ax1.plot(epoch_list, history.history['accuracy'], label = 'Train Accuracy') 
ax1.plot(epoch_list, history.history['val_accuracy'], label = 'Validation Accuracy') 
ax1.set_xticks(np.arange(1, max_epoch, 5)) 
ax1.set_ylabel('ACCCURACY VALUE') 
ax1.set_xlabel('EPOCH') 
ax1.set_title('ACCURACY') 
l1 = ax1.legend(loc = 'best') 



ax2.plot(epoch_list, history.history['loss'], label = 'Train loss') 
ax2.plot(epoch_list, history.history['val_loss'], label = 'Validation loss') 
ax2.set_xticks(np.arange(1, max_epoch, 5)) 
ax2.set_ylabel('LOSS VALUE') 
ax2.set_xlabel('EPOCH') 
ax2.set_title('LOSS') 
l2 = ax1.legend(loc = 'best') 


model.save('MALARIA_CNN.h5')





















