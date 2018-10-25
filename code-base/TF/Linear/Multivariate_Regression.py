import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
rng = np.random

# Parameters
learning_rate = 0.01
training_epochs = 10000
display_step = 50

# Read in data from csv to a dataframe
df=pd.read_csv('weather_medium.csv', sep=',')

# Reformate the type of the date/time column
df['Formatted Date'] =  pd.to_datetime(df['Formatted Date'])

# Create a new column for year / month / hour
df['Year'] = pd.DatetimeIndex(df['Formatted Date']).year
df['Month'] = pd.DatetimeIndex(df['Formatted Date']).month
df['Hour'] = pd.DatetimeIndex(df['Formatted Date']).hour

# Remove original datet/time column
df=df.drop(['Formatted Date'],axis=1)

# Convert columns to factors
df['Summary'] = df['Summary'].astype('category')
df['Precip Type'] = df['Precip Type'].astype('category')
df['Daily Summary'] = df['Daily Summary'].astype('category')

print('Head of dataframe',df.head())
# plot the data so we can see how it looks
# (output is in file graph.png)

fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(df['Temperature (C)'],df['Apparent Temperature (C)'],c='b',s=0.2, label='Data')
ax.set_xlabel('Temperature (C)')
ax.set_ylabel('Apparent Temperature (C)')
ax.set_title('Apparent Temperature vs Temperature for all data')
ax.legend(loc=2)
fig.savefig('Plot_Small_Weather_Raw.png')

fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(df['Humidity'],df['Apparent Temperature (C)'],c='b',s=0.2, label='Data')
ax.set_xlabel('Humidity')
ax.set_ylabel('Apparent Temperature (C)')
ax.set_title('Apparent Temperature vs Humidity for all data')
ax.legend(loc=2)

fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(df['Wind Speed (km/h)'],df['Apparent Temperature (C)'],c='b',s=0.2, label='Data')
ax.set_xlabel('Wind Speed (km/h)')
ax.set_ylabel('Apparent Temperature (C)')
ax.set_title('Apparent Temperature vs Wind Speed (km/h) for all data')
ax.legend(loc=2)

fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(df['Wind Bearing (degrees)'],df['Apparent Temperature (C)'],c='b',s=0.2, label='Data')
ax.set_xlabel('Wind Bearing (degrees)')
ax.set_ylabel('Apparent Temperature (C)')
ax.set_title('Apparent Temperature vs Wind Bearing (degrees) for all data')
ax.legend(loc=2)

fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(df['Visibility (km)'],df['Apparent Temperature (C)'],c='b',s=0.2, label='Data')
ax.set_xlabel('Visibility (km)')
ax.set_ylabel('Apparent Temperature (C)')
ax.set_title('Apparent Temperature vs Visibility (km) for all data')
ax.legend(loc=2)

fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(df['Loud Cover'],df['Apparent Temperature (C)'],c='b',s=0.2, label='Data')
ax.set_xlabel('Loud Cover')
ax.set_ylabel('Apparent Temperature (C)')
ax.set_title('Apparent Temperature vs Loud Cover for all data')
ax.legend(loc=2)

fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(df['Pressure (millibars)'],df['Apparent Temperature (C)'],c='b',s=0.2, label='Data')
ax.set_xlabel('Pressure (millibars)')
ax.set_ylabel('Apparent Temperature (C)')
ax.set_title('Apparent Temperature vs Pressure (millibars) for all data')
ax.legend(loc=2)

fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(df['Year'],df['Apparent Temperature (C)'],c='b',s=0.2, label='Data')
ax.set_xlabel('Year')
ax.set_ylabel('Apparent Temperature (C)')
ax.set_title('Apparent Temperature vs Year for all data')
ax.legend(loc=2)

fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(df['Month'],df['Apparent Temperature (C)'],c='b',s=0.2, label='Data')
ax.set_xlabel('Month')
ax.set_ylabel('Apparent Temperature (C)')
ax.set_title('Apparent Temperature vs Month for all data')
ax.legend(loc=2)

fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(df['Hour'],df['Apparent Temperature (C)'],c='b',s=0.2, label='Data')
ax.set_xlabel('Hour')
ax.set_ylabel('Apparent Temperature (C)')
ax.set_title('Apparent Temperature vs Hour for all data')
ax.legend(loc=2)

# df.dtypes
y = df['Apparent Temperature (C)']
X = df['Temperature (C)']

#print(X.head(10))
#print(y.head(10))

Zipped_data = pd.concat([X, y], axis=1)
#print(Zipped_data)

msk = np.random.rand(len(Zipped_data)) < 0.8
train = Zipped_data[msk]
test = Zipped_data[~msk]



# Training Data
train_X = train.values[0]
train_Y = train.values[1]
n_samples = train_X.shape[0]

# tf Graph Input
X = tf.placeholder("float")
Y = tf.placeholder("float")

# Set model weights
W = tf.Variable(rng.randn(), name="weight")
b = tf.Variable(rng.randn(), name="bias")

# Construct a linear model
pred = tf.add(tf.multiply(X, W), b)

# Mean squared error
cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)
# Gradient descent
#  Note, minimize() knows to modify W and b because Variable objects are trainable=True by default
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
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

    # Graphic display
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.savefig("lrm1")

    # Testing example, as requested (Issue #2)
    test_X = np.asarray([6.83, 4.668, 8.9, 7.91, 5.7, 8.7, 3.1, 2.1])
    test_Y = np.asarray([1.84, 2.273, 3.2, 2.831, 2.92, 3.24, 1.35, 1.03])

    print("Testing... (Mean square loss Comparison)")
    testing_cost = sess.run(
        tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * test_X.shape[0]),
        feed_dict={X: test_X, Y: test_Y})  # same function as cost above
    print("Testing cost=", testing_cost)
    print("Absolute mean square loss difference:", abs(
        training_cost - testing_cost))

    plt.plot(test_X, test_Y, 'bo', label='Testing data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.savefig("lrm")