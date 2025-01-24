#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

pd.set_option("display.max_rows",None)
pd.set_option("display.max_columns",None)


# In[24]:


fn=pd.read_csv(r"E:\HRAnalytics\HR Analytics\Final excel sheet.csv")


# In[25]:


fn.shape


# In[26]:


fn.head()


# In[27]:


for column in fn.columns:
    fn[column]=fn[column].replace({'-':np.nan})
   


# In[28]:


fn=fn.drop(['Termination Date'],axis=1)


# In[29]:


fn.head()


# In[30]:


fn.shape


# In[31]:


fn.isnull().sum()


# In[32]:


fn['Utilization%']=fn['Utilization%'].str.replace('%',' ')


# In[33]:


fn.info()


# In[34]:


fn['Current Status'].replace({'Active':0,'Resigned':1},inplace=True)


# In[35]:


fn['Current Status'].value_counts()


# In[36]:


fn.info()


# In[37]:


name=fn['Employee Name']


# In[38]:


fn=fn.drop(['Employee Name'],axis=1)


# In[39]:


fn=fn.drop(['Employee No'],axis=1)


# In[40]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[41]:


fn['Utilization%']=fn['Utilization%'].astype(float)


# In[42]:


fn.info()


# In[43]:


fn[fn.select_dtypes(include=['object']).columns] = fn[fn.select_dtypes(include=['object']).columns].apply(le.fit_transform)


# In[44]:


fn.head()


# In[61]:


active_test=fn[fn['Current Status']==0]
active_test


# In[62]:


active_test.shape


# In[63]:


resigned_train=fn[fn['Current Status']==1]
resigned_train


# In[64]:


resigned_train.shape


# In[65]:


from sklearn.model_selection import train_test_split


# In[68]:


active_test_y=active_test.iloc[:,-1]


# In[70]:


active_test_y.head()


# In[74]:


active_test_x=active_test.iloc[:,0:-1]


# In[75]:


active_test_x.shape


# In[77]:


active_test_x.head()


# In[76]:


resigned_train_x=resigned_train.iloc[:,0:-1]
resigned_train_y=resigned_train.iloc[:,-1]


# In[78]:


resigned_train_x.shape


# In[79]:


resigned_train_y.shape


# In[80]:


resigned_train_y.head()


# In[81]:


resigned_train_x.head()


# In[82]:


# logisctic


# In[85]:


from sklearn.linear_model import LogisticRegression
logreg_ar=LogisticRegression()


# In[86]:


logreg_ar.fit(resigned_train_x,resigned_train_y)


# In[ ]:




