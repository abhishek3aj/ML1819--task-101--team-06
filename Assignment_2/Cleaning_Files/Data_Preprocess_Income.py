
# coding: utf-8

# In[10]:


# Import Necessary Libraries
import pandas as pd
import numpy as np
import os
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
import time
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer


# In[11]:


# Set Working Directory
os.getcwd() 
# os.chdir() 


# In[12]:


# Read in data from Cancer.csv to a dataframe
income_data=pd.read_csv('Income.csv', sep=',')
# print("Income Data shape",income_data.shape)
# print("Income Data Types \n",income_data.dtypes)
# print("Income Data Head \n",income_data.head)

# Remove incomplete values
income_data=income_data.dropna()
# print("Income Data shape",income_data.shape)


# In[13]:


# One hot encode the categorical variables and assign to new data set
income_data_new=pd.get_dummies(income_data, columns=["workclass", "education","marital-status","occupation","relationship",
                                                    "race","sex","native-country"], 
                   prefix=["workclass", "education","marital-status","occupation","relationship",
                          "race","sex","native-country"])
# print("Income Data shape",income_data_new.shape)
# print("Income Data shape",income_data_new.dtypes)


# In[14]:


# convert target variable
income_data_new.loc[income_data_new.target == '<=50K', 'target'] = 0  
income_data_new.loc[income_data_new.target == '>50K', 'target'] = 1  


# In[15]:


# scale all continous variables
income_data_new["age"]=scale(income_data_new["age"])
income_data_new["fnlwgt"]=scale(income_data_new["fnlwgt"])
income_data_new["education-num"]=scale(income_data_new["education-num"])
income_data_new["capital-gain"]=scale(income_data_new["capital-gain"])
income_data_new["capital-loss"]=scale(income_data_new["capital-loss"])
income_data_new["hours-per-week"]=scale(income_data_new["hours-per-week"])


# In[16]:


income_data_new.to_csv('income_clean.csv', encoding='utf-8', index=False)

