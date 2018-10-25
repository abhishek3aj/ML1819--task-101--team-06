
# coding: utf-8

# In[38]:


import pandas as pd
import numpy as np


# In[39]:


### Function to encode cyclical continous variables
def encode(data, col, max_val):
    data[col + '_sin'] = np.sin(2 * np.pi * data[col]/max_val)
    data[col + '_cos'] = np.cos(2 * np.pi * data[col]/max_val)
    return data


# In[40]:


# Read in data from small csv to a dataframe
df1=pd.read_csv('weather_small.csv', sep=',')

# Reformat data in date/time column 
df1['Formatted Date'] =  pd.to_datetime(df1['Formatted Date'])

# Create a new column for year / month / hour
df1['Year'] = pd.DatetimeIndex(df1['Formatted Date']).year
df1['Month'] = pd.DatetimeIndex(df1['Formatted Date']).month
df1['Hour'] = pd.DatetimeIndex(df1['Formatted Date']).hour

# Encode month, hour and wind bearing for cyclical nature
df1 = encode(df1, 'Month', 12)
df1 = encode(df1, 'Hour', 23)
df1 = encode(df1, 'Wind Bearing (degrees)', 359)

# Remove original date/time column
df1=df1.drop(['Formatted Date'],axis=1)

# Convert columns to factors
df1['Summary'] = df1['Summary'].astype('category')
df1['Precip Type'] = df1['Precip Type'].astype('category')
df1['Daily Summary'] = df1['Daily Summary'].astype('category')

# a=set((df1.Summary.unique()))
# print(a)
# b=df1.Summary.value_counts()
# print(b)
# Create a column stating whether its mostly cloudy / overcast or not in summary
df1['Heavy_Cloud'] = pd.np.where(df1.Summary.str.contains("Mostly Cloudy"), 1,
                    pd.np.where(df1.Summary.str.contains("Overcast"), 1,
                    pd.np.where(df1.Summary.str.contains("Foggy"), 1,0)))

# Convert to boolean and print count
df1['Heavy_Cloud']=df1['Heavy_Cloud'].astype('bool')
# print(df1['Cloud_Cover'].value_counts())

# Create a column stating whether its mostly cloudy / overcast or not in summary
df1['New_Summary'] = pd.np.where(df1.Summary.str.contains("Mostly Cloudy"), "Heavy Cloud / Rain",
                    pd.np.where(df1.Summary.str.contains("Overcast"), "Heavy Cloud / Rain",
                    pd.np.where(df1.Summary.str.contains("Foggy"), "Heavy Cloud / Rain",
                    pd.np.where(df1.Summary.str.contains("Rain"), "Heavy Cloud / Rain",
                    pd.np.where(df1.Summary.str.contains("Drizzle"), "Heavy Cloud / Rain",            
                    pd.np.where(df1.Summary.str.contains("Partly Cloudy"), "Light Cloud","Clear"))))))

# Convert to boolean and print count
df1['New_Summary']=df1['New_Summary'].astype('category')
print(df1.New_Summary.value_counts())


# In[41]:


# Read in data from medium csv to a dataframe
df2=pd.read_csv('weather_medium.csv', sep=',')

# Reformate the type of the date/time column 
df2['Formatted Date'] =  pd.to_datetime(df2['Formatted Date'])

# Create a new column for year / month / hour
df2['Year'] = pd.DatetimeIndex(df2['Formatted Date']).year
df2['Month'] = pd.DatetimeIndex(df2['Formatted Date']).month
df2['Hour'] = pd.DatetimeIndex(df2['Formatted Date']).hour

# Encode month, hour and wind bearing for cyclical nature
df2 = encode(df2, 'Month', 12)
df2 = encode(df2, 'Hour', 23)
df2 = encode(df2, 'Wind Bearing (degrees)', 359)

# Remove original datet/time column
df2=df2.drop(['Formatted Date'],axis=1)

# Convert columns to factors
df2['Summary'] = df2['Summary'].astype('category')
df2['Precip Type'] = df2['Precip Type'].astype('category')
df2['Daily Summary'] = df2['Daily Summary'].astype('category')

# Create a column stating whether its mostly cloudy / overcast or not in summary
df2['Heavy_Cloud'] = pd.np.where(df2.Summary.str.contains("Mostly Cloudy"), 1,
                    pd.np.where(df2.Summary.str.contains("Overcast"), 1,
                    pd.np.where(df2.Summary.str.contains("Foggy"), 1,0)))

# Convert to boolean and print count
df2['Heavy_Cloud']=df2['Heavy_Cloud'].astype('bool')
# print(df1['Cloud_Cover'].value_counts())

# Create a column stating whether its mostly cloudy / overcast or not in summary
df2['New_Summary'] = pd.np.where(df2.Summary.str.contains("Mostly Cloudy"), "Heavy Cloud / Rain",
                    pd.np.where(df2.Summary.str.contains("Overcast"), "Heavy Cloud / Rain",
                    pd.np.where(df2.Summary.str.contains("Foggy"), "Heavy Cloud / Rain",
                    pd.np.where(df2.Summary.str.contains("Rain"), "Heavy Cloud / Rain",
                    pd.np.where(df2.Summary.str.contains("Drizzle"), "Heavy Cloud / Rain",            
                    pd.np.where(df2.Summary.str.contains("Partly Cloudy"), "Light Cloud","Clear"))))))

# Convert to boolean and print count
df2['New_Summary']=df2['New_Summary'].astype('category')
print(df1.New_Summary.value_counts())


# In[42]:


# Read in data from large csv to a dataframe
df3=pd.read_csv('weather_large.csv', sep=',')

# Reformate the type of the date/time column 
df3['Formatted Date'] =  pd.to_datetime(df3['Formatted Date'])

# Create a new column for year / month / hour
df3['Year'] = pd.DatetimeIndex(df3['Formatted Date']).year
df3['Month'] = pd.DatetimeIndex(df3['Formatted Date']).month
df3['Hour'] = pd.DatetimeIndex(df3['Formatted Date']).hour

# Encode month, hour and wind bearing for cyclical nature
df3 = encode(df3, 'Month', 12)
df3 = encode(df3, 'Hour', 23)
df3 = encode(df3, 'Wind Bearing (degrees)', 359)

# Remove original datet/time column
df3=df3.drop(['Formatted Date'],axis=1)

# Convert columns to factors
df3['Summary'] = df3['Summary'].astype('category')
df3['Precip Type'] = df3['Precip Type'].astype('category')
df3['Daily Summary'] = df3['Daily Summary'].astype('category')

# Create a column stating whether its mostly cloudy / overcast or not in summary
df3['Heavy_Cloud'] = pd.np.where(df3.Summary.str.contains("Mostly Cloudy"), 1,
                    pd.np.where(df3.Summary.str.contains("Overcast"), 1,
                    pd.np.where(df3.Summary.str.contains("Foggy"), 1,0)))

# Convert to boolean and print count
df3['Heavy_Cloud']=df3['Heavy_Cloud'].astype('bool')
# print(df1['Cloud_Cover'].value_counts())

# Create a column stating whether its mostly cloudy / overcast or not in summary
df3['New_Summary'] = pd.np.where(df3.Summary.str.contains("Mostly Cloudy"), "Heavy Cloud / Rain",
                    pd.np.where(df3.Summary.str.contains("Overcast"), "Heavy Cloud / Rain",
                    pd.np.where(df3.Summary.str.contains("Foggy"), "Heavy Cloud / Rain",
                    pd.np.where(df3.Summary.str.contains("Rain"), "Heavy Cloud / Rain",
                    pd.np.where(df3.Summary.str.contains("Drizzle"), "Heavy Cloud / Rain",            
                    pd.np.where(df3.Summary.str.contains("Partly Cloudy"), "Light Cloud","Clear"))))))

# Convert to boolean and print count
df3['New_Summary']=df3['New_Summary'].astype('category')
print(df3.New_Summary.value_counts())


# In[46]:


# Read in data from Full csv to a dataframe
df=pd.read_csv('weatherHistory.csv', sep=',')

# Reformate the type of the date/time column 
df['Formatted Date'] =  pd.to_datetime(df['Formatted Date'])

# Create a new column for year / month / hour
df['Year'] = pd.DatetimeIndex(df['Formatted Date']).year
df['Month'] = pd.DatetimeIndex(df['Formatted Date']).month
df['Hour'] = pd.DatetimeIndex(df['Formatted Date']).hour

# Encode month, hour and wind bearing for cyclical nature
df = encode(df, 'Month', 12)
df = encode(df, 'Hour', 23)
df = encode(df, 'Wind Bearing (degrees)', 359)

# Remove original date/time column
df=df.drop(['Formatted Date'],axis=1)

# Convert columns to factors
df['Summary'] = df['Summary'].astype('category')
df['Precip Type'] = df['Precip Type'].astype('category')
df['Daily Summary'] = df['Daily Summary'].astype('category')

# Create a column stating whether its mostly cloudy / overcast or not in summary
df['Heavy_Cloud'] = pd.np.where(df.Summary.str.contains("Mostly Cloudy"), 1,
                    pd.np.where(df.Summary.str.contains("Overcast"), 1,
                    pd.np.where(df.Summary.str.contains("Foggy"), 1,0)))

# Convert to boolean
df['Heavy_Cloud']=df['Heavy_Cloud'].astype('bool')

# Create a column stating whether its mostly cloudy / overcast or not in summary
df['New_Summary'] = pd.np.where(df.Summary.str.contains("Mostly Cloudy"), "Heavy Cloud / Rain",
                    pd.np.where(df.Summary.str.contains("Overcast"), "Heavy Cloud / Rain",
                    pd.np.where(df.Summary.str.contains("Foggy"), "Heavy Cloud / Rain",
                    pd.np.where(df.Summary.str.contains("Rain"), "Heavy Cloud / Rain",
                    pd.np.where(df.Summary.str.contains("Drizzle"), "Heavy Cloud / Rain",            
                    pd.np.where(df.Summary.str.contains("Partly Cloudy"), "Light Cloud","Clear"))))))

# Convert to boolean and print count
df['New_Summary']=df['New_Summary'].astype('category')
print(df.New_Summary.value_counts())


df.to_csv('New_Weather.csv')


# In[44]:


print(df1.shape)
print(df2.shape)
print(df3.shape)

