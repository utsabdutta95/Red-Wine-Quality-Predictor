#!/usr/bin/env python
# coding: utf-8

# In[3]:


#importing the libraries
import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns


# In[5]:


#reading data

df=pd.read_csv('red_wine_quality.csv')
df


# In[6]:


#To get better understanding of the data

# See the number of rows and columns
print("Rows, columns: " + str(df.shape))

# See the first five rows of the dataset
df.head()


# In[7]:


#Let's see if there are any missing values
print(df.isna().sum())


# In[12]:


#we can note that most wies have ph between 3.0 to 4.0
df['pH'].unique()


# In[14]:


#In summarizing the data, we can see that the residual sugar has a huge outlier from the max of 15.5 which is quite far from the mean of 2.5 with a median (50%) of 2.2. These differences can also be seen in the free_sulfur_dioxide , total_sulfur_dioxide, sulphates, alcohol
df.describe()


# In[17]:


corr = df.corr()
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, cmap=sns.diverging_palette(220, 20, as_cmap=True))


# In[19]:


# Create Classification version of target variable
df['goodquality'] = [1 if x >= 7 else 0 for x in df['quality']]

# Separate feature variables and target variable
X = df.drop(['quality','goodquality'], axis = 1)
y = df['goodquality']


# In[20]:


# See proportion of good vs bad wines
df['goodquality'].value_counts()


# In[21]:


#PREPARING DATA FOR DATA MODELLING


# In[22]:


## Normalize feature variables
from sklearn.preprocessing import StandardScaler
X_features = X
X = StandardScaler().fit_transform(X)


# In[24]:


#TRAINING MY DATASET
# Splitting the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=0)


# ---> For this project, I wanted to compare five different machine learning models: decision trees, random forests, AdaBoost, Gradient Boost, and XGBoost. For the purpose of this project, I wanted to compare these models by their accuracy.
# 
# 

# In[25]:


#Modelling
#using Decision Tree

from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
model1 = DecisionTreeClassifier(random_state=1)
model1.fit(X_train, y_train)
y_pred1 = model1.predict(X_test)
print(classification_report(y_test, y_pred1))


# In[27]:


# Model 2, Random Forest 
from sklearn.ensemble import RandomForestClassifier
model2 = RandomForestClassifier(random_state=1)
model2.fit(X_train, y_train)
y_pred2 = model2.predict(X_test)
print(classification_report(y_test, y_pred2))


# In[28]:


#Model 3 Adaboost
from sklearn.ensemble import AdaBoostClassifier
model3 = AdaBoostClassifier(random_state=1)
model3.fit(X_train, y_train)
y_pred3 = model3.predict(X_test)
print(classification_report(y_test, y_pred3))


# In[29]:


#Model 4 gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier
model4 = GradientBoostingClassifier(random_state=1)
model4.fit(X_train, y_train)
y_pred4 = model4.predict(X_test)
print(classification_report(y_test, y_pred4))


# In[32]:


#Model 5 XG BOOST
import xgboost as xgb
model5 = xgb.XGBClassifier(random_state=1)
model5.fit(X_train, y_train)
y_pred5 = model5.predict(X_test)
print(classification_report(y_test, y_pred5))


# By comparing the five models, the random forest and XGBoost seems to yield the highest level of accuracy. However, since XGBoost has a better f1-score for predicting good quality wines (1), I’m concluding that the XGBoost is the winner of the five models.

# In[34]:


#feature Importance via random forest
feat_importances = pd.Series(model2.feature_importances_, index=X_features.columns)
feat_importances.nlargest(25).plot(kind='barh',figsize=(10,10))


# In[35]:


#via xgboost
feat_importances = pd.Series(model5.feature_importances_, index=X_features.columns)
feat_importances.nlargest(25).plot(kind='barh',figsize=(10,10))


# In[36]:


#Comparing the Top 4 Features


# In[38]:


# Filtering df for only good quality
df_temp = df[df['goodquality']==1]
df_temp.describe()


# In[39]:


# Filtering df for only bad quality
df_temp2 = df[df['goodquality']==0]
df_temp2.describe()


# In[40]:





# In[ ]:




