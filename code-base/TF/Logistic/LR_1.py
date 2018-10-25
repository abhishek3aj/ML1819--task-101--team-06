#!/usr/bin/env python
# coding: utf-8

# In[30]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import tensorflow as tf
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt
from sklearn.preprocessing import label_binarize
from scipy import interp


# In[6]:


print(tf.__version__)


# In[137]:


weather = pd.read_csv("/home/abhishek/Documents/Ml-project/TCD-ML-18-19-Team-6-/code-base/weather_small.csv")
weather.head(5)


# In[138]:


li = list(set(weather.New_Summary))
li


# In[139]:


i = 0
summaryMap = {}
for l in li:
    summaryMap[l] = i
    i = i+1
print(summaryMap)


# In[140]:


#converting data New_Summary to numeric class
weather.New_Summary = weather.New_Summary.replace(to_replace=list(summaryMap.keys()), value=summaryMap.values())
weather.head(5)


# In[141]:


# Creating input and prediction variable
Y = weather.New_Summary.values
Y_new = np.zeros((len(Y), 3))
for i in range(0, len(Y)):
    elem = Y[i]
    arr = [0,0,0]
    arr[elem] = 1
    Y_new[i] = arr 

X = weather.drop(labels=['Summary', 'Precip Type', 'Wind Bearing (degrees)', 'Loud Cover', 'Pressure (millibars)',
                        'Daily Summary', 'Year', 'Month', 'Hour', 'Month_sin', 'Hour_sin','Hour_cos','Wind Bearing (degrees)_sin',
                        'Wind Bearing (degrees)_cos', 'Heavy_Cloud','New_Summary'],
              axis=1).values


# In[142]:


seed = 5
np.random.seed(seed)
tf.set_random_seed(seed)


# In[143]:


# Divining data into test and training set
train_index = np.random.choice(len(X), round(len(X) * 0.75), replace=False)
test_index = np.array(list(set(range(len(X))) - set(train_index)))


# In[144]:


train_X = X[train_index]
train_Y = Y_new[train_index]
test_X = X[test_index]
test_Y = Y_new[test_index]


# In[145]:


#Normalizing Data
def normalizedFunc(data):
    col_max = np.max(data, axis=0)
    col_min = np.min(data, axis=0)
    return np.divide(data - col_min, col_max - col_min)

train_X = normalizedFunc(train_X)

test_X = normalizedFunc(test_X)


# In[146]:


# Declare feature for model and hyperparameter
batch_size = 1000
W = tf.Variable(tf.zeros([6, 3]))
b = tf.Variable(tf.zeros([3]))

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
# Variable for model
data = tf.placeholder(dtype=tf.float32, shape=[None, 6])
target = tf.placeholder(dtype=tf.float32, shape=[None, 3])

mod = tf.nn.softmax(tf.matmul(data, W) + b)
loss = loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=mod, labels=target))

learning_rate = 0.01

iter_num = 10000
opt = tf.train.GradientDescentOptimizer(learning_rate)

goal = opt.minimize(loss)

prediction = tf.round(mod)
correct = tf.cast(tf.equal(prediction, target), dtype=tf.float32)
accuracy =  tf.reduce_mean(correct)


# In[147]:


loss_trace = []
train_acc = []
test_acc = []
# training model
for epoch in range(iter_num):
    # Generate random batch index
    batch_index = np.random.choice(len(train_X), size=batch_size)
    batch_train_X = train_X[batch_index]
    batch_train_y = train_Y[batch_index]
    sess.run(goal, feed_dict={data: batch_train_X, target: batch_train_y})
    temp_loss = sess.run(loss, feed_dict={data: batch_train_X, target: batch_train_y})
    # convert into a matrix, and the shape of the placeholder to correspond
    temp_train_acc = sess.run(accuracy, feed_dict={data: train_X, target: train_Y})
    temp_test_acc = sess.run(accuracy, feed_dict={data: test_X, target: test_Y})
    # recode the resultx
    loss_trace.append(temp_loss)
    train_acc.append(temp_train_acc)
    test_acc.append(temp_test_acc)

    if (epoch + 1) % 300 == 0:
        print('epoch: {:4d} loss: {:5f} train_acc: {:5f} test_acc: {:5f}'.format(epoch + 1, temp_loss,
                                                                          temp_train_acc, temp_test_acc))


# In[148]:



plt.plot(loss_trace)
plt.title('Cross Entropy Loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

plt.plot(train_acc, 'b-', label='train accuracy')
plt.plot(test_acc, 'k-', label='test accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Train and Test Accuracy')
plt.legend(loc='best')
plt.show()


# In[149]:


# Confusion Matrix
feed_dict = {data: X}
#print(X)
classification = sess.run(tf.argmax(mod, -1), feed_dict=feed_dict)

cm = metrics.confusion_matrix(Y, classification)
plt.figure(figsize=(10,8))
plt.imshow(cm, interpolation='nearest', cmap='Pastel1')
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
width, height = cm.shape
for x in range(width):
    for y in range(height):
        plt.annotate(str(cm[x][y]), xy=(y, x), 
        horizontalalignment='centre',
        verticalalignment='center')

plt.subplots_adjust(top=0.9, bottom=0.1, left=0.125, right=0.9, hspace=0.7 ,wspace=0.5)
plt.savefig('Logistic_Regression_Validation_Confusion_Small.png',pdi=600)


# In[150]:


y_valid_bin = label_binarize(Y, classes=[2,1,0])
predictions_bin=label_binarize(classification,classes=[2,1,0])
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


# In[151]:


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

