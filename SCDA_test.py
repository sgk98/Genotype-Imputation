#!/usr/bin/env python
# coding: utf-8

# <h1>Sparse Convolutional Denoising Autoencoders for Genotype Imputation <span class="tocSkip"></span></h1>

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#Dataset" data-toc-modified-id="Dataset-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Dataset</a></span><ul class="toc-item"><li><span><a href="#Loading-data" data-toc-modified-id="Loading-data-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Loading data</a></span></li><li><span><a href="#Preprocessing" data-toc-modified-id="Preprocessing-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Preprocessing</a></span></li></ul></li><li><span><a href="#Method" data-toc-modified-id="Method-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Method</a></span><ul class="toc-item"><li><span><a href="#Load-model" data-toc-modified-id="Load-model-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>Load model</a></span></li><li><span><a href="#Prediction-on-test-data" data-toc-modified-id="Prediction-on-test-data-3.2"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>Prediction on test data</a></span></li></ul></li></ul></div>

# # Introduction
# 
# This notebook demonstrates a case study of testing a SCDA model on yeast genotype dataset with 10% missing genotypes. 

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

#from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, UpSampling1D, Dropout
from keras.regularizers import l1
from keras.utils import to_categorical

from keras.models import load_model


# # Dataset

# ## Loading data

# In[2]:


input_name = 'data/yeast_genotype_test.txt'
df_ori = pd.read_csv(input_name, sep='\t', index_col=0)
df_ori.shape


# In[3]:


df_ori.head()


# ## Preprocessing

# In[4]:


# one hot encode
test_X = to_categorical(df_ori)
test_X.shape


# # Method

# ## Load model

# In[5]:


# returns a compiled model
SCDA = load_model('model/SCDA.h5')


# ## Prediction on test data

# In[6]:


# hyperparameters
missing_perc = 0.1


# In[7]:


test_X_missing = test_X.copy()
test_X_missing.shape


# In[8]:


def cal_prob(predict_missing_onehot):
    # calcaulate the probility of genotype 0, 1, 2
    predict_prob = predict_missing_onehot[:,:,1:3] / predict_missing_onehot[:,:,1:3].sum(axis=2, keepdims=True)
    return predict_prob[0]


# In[9]:


avg_accuracy = []
for i in range(test_X_missing.shape[0]):
    # Generates missing genotypes
    missing_size = int(missing_perc * test_X_missing.shape[1])
    missing_index = np.random.randint(test_X_missing.shape[1],
                                      size=missing_size)
    test_X_missing[i, missing_index, :] = [1, 0, 0]

    # predict
    predict_onehot = SCDA.predict(test_X_missing[i:i + 1, :, :])
    # only care the missing position
    predict_missing_onehot = predict_onehot[0:1, missing_index, :]
    
    # calculate probability and save file.
    predict_prob = cal_prob(predict_missing_onehot)
    pd.DataFrame(predict_prob).to_csv('results/{}.csv'.format(df_ori.index[i]),
                                      header=[1, 2],
                                      index=False)

    # predict label
    predict_missing = np.argmax(predict_missing_onehot, axis=2)
    # real label
    label_missing_onehot = test_X[i:i + 1, missing_index, :]
    label_missing = np.argmax(label_missing_onehot, axis=2)
    # accuracy
    correct_prediction = np.equal(predict_missing, label_missing)
    accuracy = np.mean(correct_prediction)
    print('{}/{}, sample ID: {}, accuracy: {:.4f}'.format(
        i, test_X_missing.shape[0], df_ori.index[i], accuracy))

    avg_accuracy.append(accuracy)


# In[10]:


print('The average imputation accuracy'       'on test data with {} missing genotypes is {:.4f}: '
    .format(missing_perc, np.mean(avg_accuracy)))


# In[ ]:




