
# coding: utf-8

# In[18]:


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


# In[19]:


# Set Working Directory
os.getcwd() 
# os.chdir() 


# In[20]:


# Read in data from Cancer.csv to a dataframe
cancer_data=pd.read_csv('Cancer.csv', sep=',')

# print("Cancer Data shape",cancer_data.shape)
# print("Cancer Data Types \n",cancer_data.dtypes)
# print("Cancer Data Head \n",cancer_data.head)


# In[21]:


# convert target variable
cancer_data.loc[cancer_data.Diagnosis == 'B', 'Diagnosis'] = 0  
cancer_data.loc[cancer_data.Diagnosis == 'M', 'Diagnosis'] = 1  


# In[22]:


# Scale variables to be used
cancer_data_scaled=pd.DataFrame(scale(pd.DataFrame(cancer_data.iloc[:,2:32])))
cancer_data_scaled = pd.concat([pd.DataFrame(cancer_data.iloc[:,0:2]), cancer_data_scaled], axis=1)
cancer_data_scaled.columns=list(cancer_data)
# print("Cancer Data Scaled shape",cancer_data_scaled.shape)
# print("Cancer Data Scaled Types \n",cancer_data_scaled.dtypes)
# print("Cancer Data Head Scaled \n",cancer_data_scaled[:6])


# In[23]:


cancer_data_scaled.to_csv('cancer_clean.csv', encoding='utf-8', index=False)

