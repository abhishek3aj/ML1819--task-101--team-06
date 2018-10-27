#!/usr/bin/env python
# coding: utf-8

# In[27]:


from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
from random import choices
rng = np.random
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import tensorflow as tf
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt
from sklearn.preprocessing import label_binarize
from scipy import interp
import itertools		
print(tf.__version__)


# In[72]:


weather = pd.read_csv("/home/abhishek/Documents/Ml-project/TCD-ML-18-19-Team-6-/code-base/weather_small.csv")
weather = weather.rename(index=str, columns={"Apparent Temperature (C)": "Apparent Temperature", "Temperature (C)": "Temperature"})
Y = weather["Apparent Temperature"].values
X = weather["Temperature"].values
print(type(Y))
plt.plot(X, Y, 'b')
plt.show()


# In[73]:


train = weather.sample(frac = 0.8)
test = weather.drop(train.index).sample(frac = 0.5)
validate =weather.drop(train.index).drop(test.index)


# In[74]:


FEATURES = ["Temperature"]
LABEL = "Apparent Temperature"
feature_cols = [tf.feature_column.numeric_column(k) for k in FEATURES]
estimator = tf.estimator.LinearRegressor(    
        feature_columns=feature_cols,   
        model_dir="train")
def get_input_fn(data_set, num_epochs=None, n_batch = 128, shuffle=True):    
         return tf.estimator.inputs.pandas_input_fn(       
         x=pd.DataFrame({k: data_set[k].values for k in FEATURES}),       
         y = pd.Series(data_set[LABEL].values),       
         batch_size=n_batch,          
         num_epochs=num_epochs,       
         shuffle=shuffle)


# In[75]:


estimator.train(input_fn=get_input_fn(train,                                       
                                           num_epochs=None,                                      
                                           n_batch = 128,                                      
                                           shuffle=False),                                      
                                           steps=1000)


# In[76]:


ev = estimator.evaluate(    
          input_fn=get_input_fn(test,                          
          num_epochs=1,                          
          n_batch = 128,                          
          shuffle=False))


# In[77]:


y = estimator.predict(    
         input_fn=get_input_fn(validate,                          
         num_epochs=1,                          
         n_batch = 128,                          
         shuffle=False))
predictions = list(p["predictions"] for p in itertools.islice(y, len(validate)))
#print("Predictions: {}".format(str(predictions)))
pred = []
for i in predictions:
    pred.append(i[0])
plt.plot(np.asarray(validate['Temperature'].values), np.asarray(pred), 'r')
plt.show()

