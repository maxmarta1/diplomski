# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 18:49:57 2021

@author: Marta
"""

# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import keras
import tensorflow as tf
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model



# In[2]:

num_of_classes=4
base_model=tf.keras.applications.vgg16.VGG16(weights='imagenet',include_top=False)

x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(512,activation='relu')(x) 
x=Dense(512,activation='relu')(x) 
x=Dense(256,activation='relu')(x) 
preds=Dense(num_of_classes,activation='softmax')(x) 


# In[3]:


model=Model(inputs=base_model.input,outputs=preds)



# In[4]:


for layer in model.layers[:18]:
    layer.trainable=False
for layer in model.layers[18:]:
    layer.trainable=True


# In[5]:


train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator=train_datagen.flow_from_directory('./DatasetRGB'+str(num_of_classes)+'/train/',
                                                 target_size=(224,224),
                                                 color_mode='rgb',
                                                 batch_size=32,
                                                 class_mode='categorical',
                                                 shuffle=True)

test_datagen=ImageDataGenerator(preprocessing_function=preprocess_input)
test_generator=test_datagen.flow_from_directory('./DatasetRGB'+str(num_of_classes)+'/val/',
                                                 target_size=(224,224),
                                                 color_mode='rgb',
                                                 batch_size=32,
                                                 class_mode='categorical',
                                                 shuffle=False)

# In[6]:


model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])

step_size_train=train_generator.n//train_generator.batch_size
model.fit_generator(generator=train_generator,
                   steps_per_epoch=step_size_train,
                   epochs=10)
# In[7]:
ypred=model.predict(test_generator)


if num_of_classes==2:
    M=np.zeros((2,2))
    yklasa=np.argmax(ypred,axis=1)

    M[0,0]=np.count_nonzero(yklasa[:62] == 0)
    M[0,1]=np.count_nonzero(yklasa[:62] == 1)
    M[1,1]=np.count_nonzero(yklasa[62:] == 1)
    M[1,0]=np.count_nonzero(yklasa[62:] == 0)

elif num_of_classes==3:
    M=np.zeros((3,3))
    yklasa=np.argmax(ypred,axis=1)
    
    M[0,0]=np.count_nonzero(yklasa[:46] == 0)
    M[0,1]=np.count_nonzero(yklasa[:46] == 1)
    M[0,2]=np.count_nonzero(yklasa[:46] == 2)
    M[1,1]=np.count_nonzero(yklasa[46:80] == 1)
    M[1,0]=np.count_nonzero(yklasa[46:80] == 0)
    M[1,2]=np.count_nonzero(yklasa[46:80] == 2)
    M[2,1]=np.count_nonzero(yklasa[80:] == 1)
    M[2,0]=np.count_nonzero(yklasa[80:] == 0)
    M[2,2]=np.count_nonzero(yklasa[80:] == 2)
    
else:           

    M=np.zeros((4,4))
    yklasa=np.argmax(ypred,axis=1)
    
    M[0,0]=np.count_nonzero(yklasa[:28] == 0)
    M[0,1]=np.count_nonzero(yklasa[:28] == 1)
    M[0,2]=np.count_nonzero(yklasa[:28] == 2)
    M[0,3]=np.count_nonzero(yklasa[:28] == 3)
    M[1,1]=np.count_nonzero(yklasa[28:59] == 1)
    M[1,0]=np.count_nonzero(yklasa[28:59] == 0)
    M[1,2]=np.count_nonzero(yklasa[28:59] == 2)
    M[1,3]=np.count_nonzero(yklasa[28:59] == 3)
    M[2,1]=np.count_nonzero(yklasa[59:80] == 1)
    M[2,0]=np.count_nonzero(yklasa[59:80] == 0)
    M[2,2]=np.count_nonzero(yklasa[59:80] == 2)
    M[2,3]=np.count_nonzero(yklasa[59:80] == 3)
    M[3,1]=np.count_nonzero(yklasa[80:] == 1)
    M[3,0]=np.count_nonzero(yklasa[80:] == 0)
    M[3,2]=np.count_nonzero(yklasa[80:] == 2)
    M[3,3]=np.count_nonzero(yklasa[80:] == 3)
    
print(M)            

