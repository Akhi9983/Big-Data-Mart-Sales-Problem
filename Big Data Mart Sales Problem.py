#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')

# Import the numpy and pandas package

import numpy as np
import pandas as pd

# Data Visualisation
import matplotlib.pyplot as plt 
import seaborn as sns


# In[12]:


## loading the train dataset
train = pd.read_csv("https://raw.githubusercontent.com/dsrscientist/bigdatamart_rep/57854a33a5a8eebbfbf387a6a7cb20b66bd7a2d0/bigdatamart_Train.csv") 
test = pd.read_csv("https://raw.githubusercontent.com/dsrscientist/bigdatamart_rep/master/bigdatamart_Test.csv")

train.head()


# In[13]:


train.info()


# In[14]:


train.describe()

Exploratory data analysis (EDA)
# In[18]:


# Check for duplicates
Total = train.shape[0]
Dupli = train[train['Item_Identifier'].duplicated()]
print(f'There are {len(Dupli)} duplicate IDs for {Total} total entries')

This shows that our Item_Identifier has some duplicate values. since a product can exist in more than one store it is expected for this repetition.
# In[19]:


# Join Train and Test Dataset

#Create source column to later separate the data easily
train['source']='train'
test['source']='test'
data = pd.concat([train,test], ignore_index = True)
print(train.shape, test.shape, data.shape)


# In[20]:


data.isnull().sum()/data.shape[0]*100 
#show values in percentage

Note that Item_Outlet_Sales is the target variable and contains missing values because our test data does not have the Item_Outlet_Sales column.
Nevertheless, we’ll impute the missing values in Item_Weight and Outlet_SizeImputing Missing ValuesIn our EDA section, we have seen that the Item_Weight and the Outlet_Size had missing values.
In our EDA section, we have seen that the Item_Weight and the Outlet_Size had missing values.
# #Imputing the mean for Item_Weight missing values

# In[21]:


item_avg_weight = data.pivot_table(values='Item_Weight', index='Item_Identifier')
print(item_avg_weight)


# In[22]:


def impute_weight(cols):
  Weight = cols[0]
  Identifier = cols[1]

  if pd.isnull(Weight):
    return item_avg_weight['Item_Weight'][item_avg_weight.index == Identifier]

  else:
    return Weight

print('Original #missing: %d'%sum(data['Item_Weight'].isnull()))

data['Item_Weight'] = data[['Item_Weight', 'Item_Identifier']].apply(impute_weight,axis=1).astype(float)

print('Final #missing: %d'%sum(data['Item_Weight'].isnull()))


# #Categorical Predictors

# Distribution of the variable Item_Fat_Content

# In[23]:


sns.countplot(train.Item_Fat_Content)

For Item_Fat_Content there are two possible type “Low Fat” or “Regular”. However, in our data it is written in different manner. We will Correct this.
# Distribution of the variable Item_Type

# In[24]:


sns.countplot(train.Item_Type)
plt.xticks(rotation=90)


# Distribution of the variable Outlet_Size

# In[25]:


sns.countplot(train.Outlet_Size)


# Distribution of the variable Outlet_Location_Type

# In[26]:


sns.countplot(train.Outlet_Location_Type)


# Distribution of the variable Outlet_Type

# In[27]:


sns.countplot(train.Outlet_Type)
plt.xticks(rotation=90)

There seems like Supermarket Type2 , Grocery Store and Supermarket Type3 all have low numbers of stores