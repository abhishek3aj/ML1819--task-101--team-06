
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import torch.nn as nn
import torch 
from torch.autograd import Variable 
from matplotlib import pyplot as plt
import torch.nn.functional as F
import time
from statistics import mean
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import scale
from sklearn import metrics
from sklearn.preprocessing import label_binarize
from scipy.stats import mode
from scipy import interp


# In[6]:


# Read in data from small csv to a dataframe
df1=pd.read_csv('/Users/siddyboy/Downloads/ML1819--task-101--team-06-master/Assignment_2/Data_Files/Income_clean.csv', sep=',')


# In[7]:


y = df1['target']
X=df1.drop(['target','fnlwgt','education-num'], axis=1)


# In[8]:


num = len(df1)
indices = list(range(num))
split = int(max(indices)*0.2)

valid_idx = np.random.choice(indices, size=split, replace=False)
train_idx = list(set(indices) - set(valid_idx))

x_train = np.array(X.loc[train_idx,:], dtype=np.float32)
x_train = x_train.reshape(-1, 1)

y_train = np.array(y[train_idx], dtype=np.float32)
y_train = y_train.reshape(-1, 1)

x_valid = np.array(X.loc[valid_idx,:], dtype=np.float32)
x_valid = x_valid.reshape(-1, 1)

y_valid = np.array(y[valid_idx], dtype=np.float32)
y_valid = y_valid.reshape(-1, 1)

print('train: {}% | validation: {}%'.format(round(len(y_train)/len(df1),2),
                                            round(len(y_valid)/len(df1),2)))


# In[9]:


x_train = Variable(torch.Tensor(x_train))
x_train=x_train.view(-1,102)
y_train = Variable(torch.Tensor(y_train))

x_valid = Variable(torch.Tensor(x_valid))
x_valid=x_valid.view(-1,102)


n = x_train.size(0)
m = x_valid.size(0)
d = x_train.size(1)

x_train_new = x_train.unsqueeze(1).expand(n, m, d)
x_valid_new = x_valid.unsqueeze(0).expand(n, m, d)


# In[ ]:


start_time = time.clock()
dist = torch.pow(x_train_new - x_valid_new, 2).sum(2)
val, index = dist.topk(10,dim=0, largest=False, sorted=True)
print (time.clock() - start_time, "seconds")
index=torch.transpose(index,dim0=0,dim1=1)
predictions=y_train[index]
predictions=predictions.detach().numpy()

a=mode(predictions, axis=1)
a=(np.squeeze(a))
a=np.transpose(a)
a=np.delete(a, np.s_[1], axis=1)

predictions=a


# In[ ]:


# Confusion Matrix
cm = metrics.confusion_matrix(y_valid, predictions)
plt.figure(figsize=(10,8))
plt.imshow(cm, interpolation='nearest', cmap='Pastel1')
plt.title('Confusion Matrix  knn Pytorch Validation Data', size = 12)
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ["positive", "negative"], size = 12)
plt.yticks(tick_marks, ["positive", "negative"], size = 12)
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
plt.savefig('KNN_Confusion_PT.png',pdi=600)
accuracy=sum(predictions==y_valid)/len(predictions)


# In[ ]:


ERR=1-accuracy
Recall = (cm[0,0])/(np.sum(cm,axis=1)[0])
FPR=(cm[1,0])/(np.sum(cm,axis=1)[1])
Specificity=1-FPR
FNR=1-Recall

print('Score of validation model: ',accuracy)
print('Error rate of validation model: ',ERR)
print('Recall / TPR of validation model: ',Recall)
print('FNR of validation model: ',FNR)
print('Specificity / TNR of validation model: ',Specificity)
print('FPR of validation model: ',FPR)


# In[ ]:


### calculate ROC Score
print("AUC of validation model",roc_auc_score(y_valid, predictions))
fpr, tpr, _ = metrics.roc_curve(y_valid, predictions)

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
ax1.set_title('ROC Curve Pytorch Validation Data ')
ax1.plot(fpr,tpr,color='Black',linewidth=2,label='ROC Curve')

textstr = 'AUC of validation model= $%.2f$'%(roc_auc_score(y_valid, predictions))
ax1.text(0.5,0,textstr)

fig.subplots_adjust(top=0.9, bottom=0.1, left=0.125, right=0.9, hspace=0.7 ,wspace=0.5)
fig.savefig('kNN_Validation_ROC_PT.png',pdi=600)

