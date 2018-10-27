#!/usr/bin/env python
# coding: utf-8

# In[61]:


import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt
from sklearn.preprocessing import label_binarize
from scipy import interp


# In[62]:


weather = pd.read_csv("/home/abhishek/Documents/Ml-project/TCD-ML-18-19-Team-6-/code-base/weather_small.csv")
weather.head(5)


# In[63]:


li = list(set(weather.New_Summary))
li
i = 0
summaryMap = {}
for l in li:
    summaryMap[l] = i
    i = i+1
print(summaryMap)
weather.New_Summary = weather.New_Summary.replace(to_replace=list(summaryMap.keys()), value=summaryMap.values())
weather.head(5)


# In[64]:


# Creating input and prediction variable
Y = weather.New_Summary.values
Y_new = np.zeros((len(Y), 3))
for i in range(0, len(Y)):
    elem = Y[i]
    arr = [0,0,0]
    arr[elem] = 1
    Y_new[i] = arr 

X = weather.drop(labels=['Summary', 'Precip Type', 'Wind Speed (km/h)', 'Wind Bearing (degrees)', 'Visibility (km)',
                        'Loud Cover', 'Pressure (millibars)', 'Daily Summary', 'Year', 'Month','Hour','Hour_sin',
                        'Hour_cos', 'Wind Bearing (degrees)_sin','Wind Bearing (degrees)_cos','Heavy_Cloud','New_Summary'],
              axis=1).values


# In[65]:


seed = 5
np.random.seed(seed)
tf.set_random_seed(seed)


# In[66]:


# Divining data into test and training set
train_index = np.random.choice(len(X), round(len(X) * 0.8), replace=False)
test_index = np.array(list(set(range(len(X))) - set(train_index)))


# In[67]:


Xtr = X[train_index]
Ytr = Y_new[train_index]
Xte = X[test_index]
Yte = Y_new[test_index]


# In[68]:


def normalizedFunc(data):
    col_max = np.max(data, axis=0)
    col_min = np.min(data, axis=0)
    return np.divide(data - col_min, col_max - col_min)

Xtr = normalizedFunc(Xtr)

Xte = normalizedFunc(Xte)


# In[69]:


# tf Graph Input


xtr = tf.placeholder("float", [None, 5])
xte = tf.placeholder("float", [5])

# Nearest Neighbor calculation using L1 Distance
# Calculate L1 Distance
distance = tf.reduce_sum(tf.abs(tf.add(xtr, tf.negative(xte))), reduction_indices=1)
# Prediction: Get min distance index (Nearest neighbor)
pred = tf.arg_min(distance, 0)

accuracy = 0.

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

predi=[]
truePred = []
# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # loop over test data
    for i in range(len(Xte)):
        # Get nearest neighbor
        nn_index = sess.run(pred, feed_dict={xtr: Xtr, xte: Xte[i, :]})
        # Get nearest neighbor class label and compare it to its true label
        print("Test", i, "Prediction:", np.argmax(Ytr[nn_index]), 
            "True Class:", np.argmax(Yte[i]))
        predi.append(np.argmax(Ytr[nn_index]))
        truePred.append(np.argmax(Yte[i]))
        # Calculate accuracy
        if np.argmax(Ytr[nn_index]) == np.argmax(Yte[i]):
            accuracy += 1./len(Xte)
    print("Done!")
    print("Accuracy:", accuracy)


# In[75]:


cm2 = metrics.confusion_matrix(truePred, predi)
#cm2 = metrics.confusion_matrix(y_test, predictions_test)
plt.figure(figsize=(10,8))
plt.imshow(cm2, interpolation='nearest', cmap='Pastel1')
plt.title('Confusion matrix validation', size = 12)
plt.colorbar()
#{'Light Cloud': 0, 'Heavy Cloud / Rain': 1, 'Clear': 2}

tick_marks = np.arange(3)
plt.xticks(tick_marks, ["Clear", "Heavy Cloud / Rain", "Light Cloud"], size = 10,rotation=0)
plt.yticks(tick_marks, ["Clear", "Heavy Cloud / Rain", "Light Cloud"], size = 10,rotation=90,verticalalignment='center')
# plt.axis["left"].major_ticklabels.set_ha("center")
plt.tight_layout()
plt.ylabel('Actual label', size = 12)
plt.xlabel('Predicted label', size = 12)
width, height = cm2.shape
for x in range(width):
    for y in range(height):
        plt.annotate(str(cm2[x][y]), xy=(y, x), 
        horizontalalignment='centre',
        verticalalignment='center')

plt.subplots_adjust(top=0.9, bottom=0.1, left=0.125, right=0.9, hspace=0.7 ,wspace=0.5)
plt.savefig('KNN_Test_Confusion_Matrix_Large.png',pdi=600)


# In[72]:


y_valid_bin = label_binarize(truePred, classes=[2,1,0])
predictions_bin=label_binarize(predi,classes=[2,1,0])
n_classes = y_valid_bin.shape[1]
print(y_valid_bin)
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_valid_bin[:, i], predictions_bin[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_valid_bin.ravel(), predictions_bin.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
predi
# Compute macro-average ROC curve and ROC area
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])


print("AUC of Clear validation model",roc_auc[0])
print("AUC of Light Cloud validation model",roc_auc[2])
print("AUC of Heavy Cloud / Rain validation model",roc_auc[1])
print("AUC of micro average validation model",roc_auc["micro"])
print("AUC of macro average validation model",roc_auc["macro"])


# In[73]:


# Plot all ROC curves
fig, ax1=plt.subplots(figsize=(6,6))
x1 = [0, 0]
x2 = [0,1]
y1=[0,1]
y2=[1,1]
ax1.plot(x1, y1, 'red',linewidth=0.5)
ax1.plot(x2, y2, 'red',linewidth=0.5)
ax1.plot(x2, y1, 'grey',linestyle='--',linewidth=0.5)

ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.set_title('ROC Curve Validation Data')

ax1.plot(fpr[0],tpr[0],
         label='Clear ROC curve (area = {0:0.2f})'
              ''.format(roc_auc[0]),
         color='lightblue',linewidth=1)

ax1.plot(fpr[2],tpr[2],
         label='Light Cloud ROC curve (area = {0:0.2f})'
              ''.format(roc_auc[2]),
         color='blue',linewidth=1)

ax1.plot(fpr[1],tpr[1],
         label='Heavy Cloud / Rain ROC curve (area = {0:0.2f})'
              ''.format(roc_auc[1]),
         color='black',linewidth=1)

ax1.plot(fpr[0],tpr[0],
         label='Clear ROC curve (area = {0:0.2f})'
              ''.format(roc_auc[0]),
         color='lightblue',linewidth=2)


ax1.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=1)

ax1.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=1)

ax1.legend(loc="lower right",fontsize=8)

fig.subplots_adjust(top=0.9, bottom=0.1, left=0.125, right=0.9, hspace=0.7 ,wspace=0.5)
fig.savefig('KNN_Validation_ROC_Small.png',pdi=600)

