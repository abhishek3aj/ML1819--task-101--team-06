
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import metrics
from matplotlib import pyplot as plt

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
X=df['Temperature (C)']

# Create Training&Validation / Test set - split of 70/20/10
X_intermediate, X_test, y_intermediate, y_test = train_test_split(X,y,test_size=0.1) 
X_valid, X_train, y_valid, y_train = train_test_split(X_intermediate, y_intermediate,
                                                      test_size=0.78)
# delete intermediate variables
X_intermediate, y_intermediate

print('train: {}% | validation: {}% | test {}%'.format(round(len(y_train)/len(df),2),
                                                       round(len(y_valid)/len(df),2),
                                                       round(len(y_test)/len(df),2)))

# Reshape training data so model can be fit
X_train=X_train.values.reshape(-1,1)
y_train=y_train.values.reshape(-1,1)

# Fit a lienar regression model to the training data
lm = linear_model.LinearRegression()
model = lm.fit(X_train,y_train)

# Reshape validation data so that model can be run
X_valid=X_valid.values.reshape(-1,1)
# Predictions on validation data
predictions = model.predict(X_valid)

# Plot outputs of the actual validation data vs the prediction line
fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(X_valid, y_valid, color='black', s=0.2,label='Validation Data')
ax.plot(X_valid, predictions, color = 'r', linewidth=2,label='Prediction Line')
ax.set_xlabel('Temperature (C)')
ax.set_ylabel('Apparent Temperature (C)')
ax.set_title('Apparent Temperature vs Temperature Actual Test validation data')
ax.legend(loc=2)
fig.savefig('pred_line.png')

print ('Coefficient of determination of the prediction:', model.score(X_valid, y_valid))

## Error metrics
print('Mean Absolute Error on validation data:', metrics.mean_absolute_error(y_valid, predictions))  
print('Mean Squared Error on validation data:', metrics.mean_squared_error(y_valid, predictions))  
print('Root Mean Squared Error on validation data:', np.sqrt(metrics.mean_squared_error(y_valid, predictions)))  

#############################################
# Reshape validation data so that model can be run
X_test=X_test.values.reshape(-1,1)
# Predictions on validation data
predictions_test = model.predict(X_test)

# Plot outputs of the actual validation data vs the prediction line
fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(X_valid, y_valid, color='black', s=0.2,label='Test Data')
ax.plot(X_test, predictions_test, color = 'r', linewidth=2,label='Prediction Line')
ax.set_xlabel('Temperature (C)')
ax.set_ylabel('Apparent Temperature (C)')
ax.set_title('Apparent Temperature vs Temperature Actual Test test data')
ax.legend(loc=2)
fig.savefig('pred_line.png')

print ('Coefficient of determination of the prediction on test data:', model.score(X_test, y_test))

## Error metrics
print('Mean Absolute Error on test data:', metrics.mean_absolute_error(y_test, predictions_test))  
print('Mean Squared Error on test data:', metrics.mean_squared_error(y_test, predictions_test))  
print('Root Mean Squared Error on test data:', np.sqrt(metrics.mean_squared_error(y_test, predictions_test)))  

