#!/usr/bin/env python
# coding: utf-8

# ### Import Libraries

# In[1]:


from keras.utils import np_utils
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import random 
from numpy import *
from PIL import Image


# ### Randomly split data to test and train

# In[19]:


import random
import shutil
import os

rootdir = 'D:/Data sciece SP project/Kaggle/no'
outdir = 'D:/Data sciece SP project/Kaggle/BraintumourTest/no'

ref = 1500

dirsAndFiles = {}   # here we store a structure  {folder: [file1, file2], folder2: [file2, file4] }
dirs = [x[0] for x in os.walk(rootdir)] # here we store all sub-dirs

for dir in dirs:
    dirsAndFiles[dir] = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]

for (dir, files) in dirsAndFiles.items():
    if len(files) > ref:
        for i in range(int(0.2*len(files))):  # copy 20% of files
            fe = random.choice(files)
            files.remove(fe)
            shutil.copy(os.path.join(dir, fe), outdir)
    else:                                            # copy all files
        for file in files:
             shutil.copy(os.path.join(dir, file), outdir)
                
rootdir = 'D:/Data sciece SP project/Kaggle/yes'
outdir = 'D:/Data sciece SP project/Kaggle/BraintumourTest/yes'

ref = 1500

dirsAndFiles = {}   # here we store a structure  {folder: [file1, file2], folder2: [file2, file4] }
dirs = [x[0] for x in os.walk(rootdir)] # here we store all sub-dirs

for dir in dirs:
    dirsAndFiles[dir] = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]

for (dir, files) in dirsAndFiles.items():
    if len(files) > ref:
        for i in range(int(0.2*len(files))):  # copy 20% of files
            fe = random.choice(files)
            files.remove(fe)
            shutil.copy(os.path.join(dir, fe), outdir)
    else:                                            # copy all files
        for file in files:
             shutil.copy(os.path.join(dir, file), outdir)


# In[20]:


rootdir = 'D:/Data sciece SP project/Kaggle/yes'
outdir = 'D:/Data sciece SP project/Kaggle/BraintumourTrain/yes'

ref = 1500

dirsAndFiles = {}   # here we store a structure  {folder: [file1, file2], folder2: [file2, file4] }
dirs = [x[0] for x in os.walk(rootdir)] # here we store all sub-dirs

for dir in dirs:
    dirsAndFiles[dir] = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]

for (dir, files) in dirsAndFiles.items():
    if len(files) > ref:
        for i in range(int(0.8*len(files))):  # copy 80% of files
            fe = random.choice(files)
            files.remove(fe)
            shutil.copy(os.path.join(dir, fe), outdir)
    else:                                            # copy all files
        for file in files:
             shutil.copy(os.path.join(dir, file), outdir)
                
rootdir = 'D:/Data sciece SP project/Kaggle/no'
outdir = 'D:/Data sciece SP project/Kaggle/BraintumourTrain/no'

ref = 1500

dirsAndFiles = {}   # here we store a structure  {folder: [file1, file2], folder2: [file2, file4] }
dirs = [x[0] for x in os.walk(rootdir)] # here we store all sub-dirs

for dir in dirs:
    dirsAndFiles[dir] = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]

for (dir, files) in dirsAndFiles.items():
    if len(files) > ref:
        for i in range(int(0.8*len(files))):  # copy 80% of files
            fe = random.choice(files)
            files.remove(fe)
            shutil.copy(os.path.join(dir, fe), outdir)
    else:                                            # copy all files
        for file in files:
             shutil.copy(os.path.join(dir, file), outdir)


# In[2]:


from tensorflow.keras.models import Sequential
#Convolution Layer
from tensorflow.keras.layers import Conv2D
#Mas Pooling Layer
from tensorflow.keras.layers import MaxPool2D
#Flatten
from tensorflow.keras.layers import Flatten
#ANN
from tensorflow.keras.layers import Dense


# ### Initialise the model

# In[32]:


model= Sequential()
#Add Convolution layer
#No. of filters
#Filter shape
#stride
#input shape -- shape of image
model.add(Conv2D(filters=32,kernel_size=(3,3),strides=(1,1),input_shape=(64,64,3),
                activation="relu"))

#Add Maxpooling layer
#Pool size
#strides
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

#Add Flatten layer
model.add(Flatten())

#Add fully Connected Layer or Hidden Layer--- ANN
model.add(Dense(kernel_initializer="random_uniform",activation="relu",units=250))

#Add Output Layer
model.add(Dense(kernel_initializer="random_uniform",activation="sigmoid",units=1))

#Compile the Model
model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])


# ### Data Preproessing

# In[33]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen=ImageDataGenerator(
        rescale=1./255,#Scale the pixel from 0 to 1
        shear_range=0.2,zoom_range=0.2,
        horizontal_flip=True)


# In[34]:


test_datagen=ImageDataGenerator(
        rescale=1./255)


# In[35]:


x_train=train_datagen.flow_from_directory(r'D:\Data sciece SP project\Kaggle\BraintumourTrain',
                                         target_size=(64,64),color_mode='rgb',
                                         class_mode='binary',batch_size=32)


# In[36]:


x_test=test_datagen.flow_from_directory(r'D:\Data sciece SP project\Kaggle\BraintumourTest',
                                      target_size=(64,64),color_mode='rgb',
                                         class_mode='binary',batch_size=32)


# In[37]:


len(x_train)


# In[38]:


x_train.class_indices


# In[39]:


plt.figure(figsize=(15,15))
for i,batch in enumerate(x_train,1):
    if(i==12):
        break
    plt.subplot(4,3,i)
    plt.imshow(batch[0][1])


# In[40]:


model.summary()


# In[41]:


len(x_test)


# In[42]:


#steps_per_epoch-- No.of bathes
model.fit_generator(x_train,steps_per_epoch=75,epochs=100,validation_data=x_test,validation_steps=19)


# In[43]:


model.save("cnn_braintumour.h5")


# In[ ]:




