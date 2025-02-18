import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import tensorflow as tf


activity = pd.read_csv("./train.csv")
activity_test = pd.read_csv("./test.csv")
print(activity.shape)
print(activity.head())
li = list(set(activity.Activity))
print(li)
summaryMap = {}
k = 0
for i in li:
    summaryMap[i] = k
    k = k +1
summaryMap = {'STANDING':0, 'WALKING_DOWNSTAIRS':1, 'LAYING':2, 'WALKING':1, 'SITTING':3, 'WALKING_UPSTAIRS':1}
activity.Activity = activity.Activity.replace(to_replace=list(summaryMap.keys()), value=summaryMap.values())
Y = activity.Activity.values
print(Y)
X = activity.drop(labels=['Activity', 'subject'], axis=1).values
activity_test.Activity = activity_test.Activity.replace(to_replace=list(summaryMap.keys()), value=summaryMap.values())
Y_test = activity_test.Activity.values
X_test = activity_test.drop(labels=['Activity','subject'], axis=1).values

seed = 5
np.random.seed(seed)
tf.set_random_seed(seed)

train_index = np.random.choice(len(X), round(len(X) * 1), replace=False)
test_index = np.random.choice(len(X_test), round(len(X) * 1))
print(len(test_index))
print(len(train_index))
train_X = X[train_index]
train_Y = Y[train_index]
test_X = X[test_index]
test_Y = Y[test_index]

print("columns : ",len(activity.columns))

#Function to normalize data in X
def normalizedFunc(data):
    col_max = np.max(data, axis=0)
    col_min = np.min(data, axis=0)
    return np.divide(data - col_min, col_max - col_min)


train_X = normalizedFunc(train_X)
train_Y = normalizedFunc(train_Y)

# Declare feature for model
A = tf.Variable(tf.random_normal(shape=[561, 1]))
b = tf.Variable(tf.random_normal(shape=[1, 1]))
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


# Variable for model
data = tf.placeholder(dtype=tf.float32, shape=[None, 561])
target = tf.placeholder(dtype=tf.float32, shape=[None, 1])

#Model declaration
mod = tf.nn.softmax(tf.add(tf.matmul(data, A), b))

#mod = tf.matmul(data, A) + b

#tf.nn.softmax(tf.matmul(, W) + b)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=mod, labels=target))

learning_rate = 0.01
batch_size = 600
iter_num = 10000


#Optimizer
opt = tf.train.GradientDescentOptimizer(learning_rate)

goal = opt.minimize(loss)

prediction = tf.round(tf.sigmoid(mod))
correct = tf.cast(tf.equal(prediction, target), dtype=tf.float32)
accuracy = tf.reduce_mean(correct)

loss_trace = []
train_acc = []
test_acc = []

# training model
for epoch in range(iter_num):
    # Generate random batch index
    batch_index = np.random.choice(len(train_X), size=batch_size)
    batch_train_X = train_X[batch_index]
    batch_train_y = np.matrix(train_Y[batch_index]).T
    sess.run(goal, feed_dict={data: batch_train_X, target: batch_train_y})

    temp_loss = sess.run(loss, feed_dict={data: batch_train_X, target: batch_train_y})
    # convert into a matrix, and the shape of the placeholder to correspond
    temp_train_acc = sess.run(accuracy, feed_dict={data: train_X, target: np.matrix(train_Y).T})
    temp_test_acc = sess.run(accuracy, feed_dict={data: test_X, target: np.matrix(test_Y).T})
    # recode the resultx
    loss_trace.append(temp_loss)
    train_acc.append(temp_train_acc)
    test_acc.append(temp_test_acc)
    # output
    if (epoch + 1) % 300 == 0:
        print('epoch: {:4d} loss: {:5f} train_acc: {:5f} test_acc: {:5f}'.format(epoch + 1, temp_loss,
                                                                          temp_train_acc, temp_test_acc))
writer = tf.summary.FileWriter("output", sess.graph)
writer.close


plt.plot(loss_trace)
plt.title('Cross Entropy Loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig('fig11')


plt.plot(train_acc, 'b-', label='train accuracy')
plt.plot(test_acc, 'k-', label='test accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Train and Test Accuracy')
plt.legend(loc='best')
plt.savefig('fig21')