#!/usr/bin/env python
# coding: utf-8

# In[22]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns

titan = pd.read_csv('/content/train.csv')
titan.head()


# In[4]:


titan.info()


# In[7]:


titan.shape
titan.isnull().sum()


# In[59]:


#dropping cabin column 
titan = titan.drop(columns= 'Cabin', axis = 1)

titan.head()


# In[11]:


# getting some statistical measures about the data
titan.describe()


# In[13]:


# finding the number of people survived and not survived
titan['Survived'].value_counts()


# In[17]:


# replacing the missing values in "Age" column with mean value
titan['Age'].fillna(titan['Age'].mean(), inplace=True)


# In[23]:


sns.set()


# In[24]:


# making a count plot for "Survived" column
sns.countplot('Survived', data=titan)


# In[27]:


titan['Sex'].value_counts()


# In[29]:


# making a count plot for "Sex" column
sns.countplot('Sex', data=titan)


# In[30]:


# number of survivors Gender wise
sns.countplot('Sex', hue='Survived', data=titan)


# In[31]:


# making a count plot for "Pclass" column
sns.countplot('Pclass', data=titan)


# In[32]:


sns.countplot('Pclass', hue='Survived', data=titan)


# In[ ]:





# In[47]:


titan['Sex'].value_counts()


# In[46]:


titan['Embarked'].value_counts()


# In[96]:


a = titan['Age']
print(a)


# In[91]:


titan['male']= titan['Sex']=='male'
X = titan[['Pclass', 'male', 'SibSp', 'Fare']]
y = titan['Survived']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=2)
print(X.shape, X_train.shape, X_test.shape)


# In[73]:


print(X)


# In[97]:


print(y)


# In[79]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)
lr = LogisticRegression()
lr.fit(X_train,y_train)


# In[80]:


X_train_prediction = lr.predict(X_train)
print(X_train_prediction)


# In[93]:


training_data_accuracy = accuracy_score(y_train, X_train_prediction)
print('Accuracy score of training data : ', training_data_accuracy)


# In[87]:


X_test_prediction = lr.predict(X_test)
print(X_test_prediction)


# In[92]:


test_data_accuracy = accuracy_score(y_test, X_test_prediction)
print('Accuracy score of test data : ', test_data_accuracy)

