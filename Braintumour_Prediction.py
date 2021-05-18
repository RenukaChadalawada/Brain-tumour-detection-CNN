#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tensorflow.keras.models import load_model
import numpy as np
import cv2
model=load_model('cnn_braintumour.h5')


# In[2]:


def detect(frame):
    img=cv2.resize(frame,(64,64))
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    #scaling to 0 to 1 range 
    if(np.max(img)>1):
            img = img/255.0
    img=np.array([img])
    prediction=model.predict_classes(img)
    label=["No Tumor","Tumor"]
    prediction=prediction[0][0]
    print("Prediction:",prediction)
    return label[prediction]


# In[3]:


image = cv2.imread("D:/Data sciece SP project/Kaggle/BraintumourPred/pred/pred0.jpg")


# In[4]:


image.shape


# In[5]:


detect(image)


# In[6]:


import glob
import cv2 as cv

path=glob.glob("D:/Data sciece SP project/Kaggle/BraintumourPred/pred/*.jpg")


# In[7]:


img_class=[]
for img in path:
    n=cv.imread(img)
    pred=detect(n)
    print(pred)
    img_class.append(pred)


# In[ ]:




