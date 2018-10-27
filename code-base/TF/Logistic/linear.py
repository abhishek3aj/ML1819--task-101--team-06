from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
from random import choices
rng = np.random
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt
from sklearn.preprocessing import label_binarize
from scipy import interp
print(tf.__version__)

weather = pd.read_csv("/home/abhishek/Documents/Ml-project/TCD-ML-18-19-Team-6-/code-base/weather_small.csv")
Y = weather["Apparent Temperature (C)"].values
X = weather["Temperature (C)"].values

seed = 5
np.random.seed(seed)
tf.set_random_seed(seed)
train_index = np.random.choice(len(X), round(len(X) * 0.70), replace=False)
#test_index = np.random.choice(len(X-train_index), round(X-train_index * 0.60), replace=False)
rem_index = np.array(list(set(range(len(X))) - set(train_index)))
test_index = np.random.choice(len(rem_index), round(len(rem_index) * 0.60), replace=False)
validation_index = np.array(list(set(range(len(rem_index))) - set(test_index)))
print(len(X))
print(len(train_index))
print(len(test_index))
print(len(validation_index))
train_X = X[train_index]
train_Y = Y[train_index]
test_X = X[test_index]
test_Y = Y[test_index]
validate_X = X[validation_index]
validate_Y = Y[validation_index]
# normalization
mean = train_X.mean(axis=0)
std = train_X.std(axis=0)
train_data = (train_X - mean) / std
test_data = (test_X - mean) / std

learning_rate = 0.01
training_epochs = 1000
display_step = 100
n_samples = train_X.shape[0]
X = tf.placeholder("float")
Y = tf.placeholder("float")
W = tf.Variable(rng.randn(), name="weight")
b = tf.Variable(rng.randn(), name="bias")
#model
pred = tf.add(tf.multiply(X, W), b)
cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
init = tf.global_variables_initializer()
print(train_X)
print(train_Y)


with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # Fit all training data
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
                "W=", sess.run(W), "b=", sess.run(b))

    print("Optimization Finished!")
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')

    plt.plot(test_X, test_Y, 'bo', label='Testing data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.savefig("linear")