
# coding: utf-8

# In[1]:


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


# In[2]:


# Read in data from Cancer.csv to a dataframe
income_data=pd.read_csv('Income_clean.csv', sep=',')

# print("Income Data shape",income_data.shape)
# print("Income Data Types \n",income_data.dtypes)
# print("Income Data Head \n",income_data.head)


# In[10]:


# Set X and y values
X=income_data[income_data.columns.difference(['target'])]
y=income_data['target']


# In[15]:


# Create Training & Test set - split of 80/20 split
X_test, X_train, y_test, y_train = train_test_split(X, y,test_size=0.8)

print('train: {}% | test {}%'.format(round(len(y_train)/len(income_data),2),
                                                       round(len(y_test)/len(income_data),2)))


# In[ ]:


### Now perform Machine Learning Algorithm - variables are called X_train, y_train, X_test, y_test
start_time = time.clock()

#### Put training algorithm in here so as to calculate time to train model

print (time.clock() - start_time, "seconds")

### Print coefficients of trained model
print('Coefficients',)
### Calculate predictions on validation data - call them predictions


# In[ ]:


### Confusion Matrix of model - fill in title, tick marks file name approriapetly
cm = metrics.confusion_matrix(y_test, predictions)
plt.figure(figsize=(10,8))
plt.imshow(cm, interpolation='nearest', cmap='Pastel1')
plt.title('Appropriate Title', size = 12)
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ["X", "Y"], size = 12)
plt.yticks(tick_marks, ["X", "Y"], size = 12)
plt.tight_layout()
plt.ylabel('Actual label', size = 12)
plt.xlabel('Predicted label', size = 12)
width, height = cm.shape
for x in range(width):
 for y in range(height):
  plt.annotate(str(cm[x][y]), xy=(y, x), 
  horizontalalignment='center',
  verticalalignment='center')

plt.subplots_adjust(top=0.9, bottom=0.1, left=0.125, right=0.9, hspace=0.7 ,wspace=0.5)
plt.savefig('File_Name.png',pdi=600)


# In[ ]:


### Calculate results - potentially need to figure out how to do confidence intervals on these results?
Score=lrm.score(X_valid, y_test)
Recall = (cm[0,0])/(np.sum(cm,axis=1)[0])
Specificity=1-(cm[1,0])/(np.sum(cm,axis=1)[1])

print('Score of validation model: ',Score)
print('Recall / TPR of validation model: ',Recall)
print('Specificity / TNR of validation model: ',Specificity)


# In[ ]:


### calculate ROC Score and plot AUC, Title of plot and file_name to be changed

print("AUC of validation model",roc_auc_score(y_test, predictions))
fpr, tpr, _ = metrics.roc_curve(y_test, predictions)

fig, ax1=plt.subplots(figsize=(6,6))
x1 = [0, 0]
x2 = [0,1]
y1=[0,1]
y2=[1,1]
ax1.plot(x1, y1, 'red')
ax1.plot(x2, y2, 'red')
ax1.plot(x2, y1, 'grey',linestyle='--')

ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.set_title('Appropriate Tile')
ax1.plot(fpr,tpr,color='Black',linewidth=2,label='ROC Curve')

textstr = 'AUC of validation model= $%.2f$'%(roc_auc_score(y_test, predictions))
ax1.text(0.5,0,textstr)

fig.subplots_adjust(top=0.9, bottom=0.1, left=0.125, right=0.9, hspace=0.7 ,wspace=0.5)
fig.savefig('file_Name.png',pdi=600)

