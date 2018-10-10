
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np

# Read in data from csv to a dataframe
df1=pd.read_csv('weather_small.csv', sep=',')

# Reformate the type of the date/time column 
df1['Formatted Date'] =  pd.to_datetime(df1['Formatted Date'])

# Create a new column for year / month / hour
df1['Year'] = pd.DatetimeIndex(df1['Formatted Date']).year
df1['Month'] = pd.DatetimeIndex(df1['Formatted Date']).month
df1['Hour'] = pd.DatetimeIndex(df1['Formatted Date']).hour

# Remove original datet/time column
df1=df1.drop(['Formatted Date'],axis=1)

# Convert columns to factors
df1['Summary'] = df1['Summary'].astype('category')
df1['Precip Type'] = df1['Precip Type'].astype('category')
df1['Daily Summary'] = df1['Daily Summary'].astype('category')

# Read in data from csv to a dataframe
df2=pd.read_csv('weather_medium.csv', sep=',')

# Reformate the type of the date/time column 
df2['Formatted Date'] =  pd.to_datetime(df2['Formatted Date'])

# Create a new column for year / month / hour
df2['Year'] = pd.DatetimeIndex(df2['Formatted Date']).year
df2['Month'] = pd.DatetimeIndex(df2['Formatted Date']).month
df2['Hour'] = pd.DatetimeIndex(df2['Formatted Date']).hour

# Remove original datet/time column
df2=df2.drop(['Formatted Date'],axis=1)

# Convert columns to factors
df2['Summary'] = df2['Summary'].astype('category')
df2['Precip Type'] = df2['Precip Type'].astype('category')
df2['Daily Summary'] = df2['Daily Summary'].astype('category')

# Read in data from csv to a dataframe
df3=pd.read_csv('weather_large.csv', sep=',')

# Reformate the type of the date/time column 
df3['Formatted Date'] =  pd.to_datetime(df3['Formatted Date'])

# Create a new column for year / month / hour
df3['Year'] = pd.DatetimeIndex(df3['Formatted Date']).year
df3['Month'] = pd.DatetimeIndex(df3['Formatted Date']).month
df3['Hour'] = pd.DatetimeIndex(df3['Formatted Date']).hour

# Remove original datet/time column
df3=df3.drop(['Formatted Date'],axis=1)

# Convert columns to factors
df3['Summary'] = df3['Summary'].astype('category')
df3['Precip Type'] = df3['Precip Type'].astype('category')
df3['Daily Summary'] = df3['Daily Summary'].astype('category')

print(df1.shape)
print(df2.shape)
print(df3.shape)

