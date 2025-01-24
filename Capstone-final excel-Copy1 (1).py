#!/usr/bin/env python
# coding: utf-8

# In[73]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

pd.set_option("display.max_rows",None)
pd.set_option("display.max_columns",None)


# In[74]:


fn=pd.read_csv(r"E:\HRAnalytics\HR Analytics\FINAL.csv")


# In[75]:


fn.shape


# In[76]:


fn.head()


# In[77]:


for column in fn.columns:
    fn[column]=fn[column].replace({'-':np.nan})
   


# In[78]:


fn=fn.drop(['Termination Date'],axis=1)


# In[79]:


fn.head()


# In[80]:


fn.shape


# In[81]:


fn.isnull().sum()


# In[82]:


fn['Utilization%']=fn['Utilization%'].str.replace('%',' ')


# In[83]:


fn.info()


# In[84]:


fn['Current Status'].replace({'Active':0,'Resigned':1},inplace=True)


# In[85]:


fn['Current Status'].value_counts()


# In[86]:


fn.info()


# In[87]:


name=fn['Employee Name']


# In[88]:


fn=fn.drop(['Employee Name'],axis=1)


# In[89]:


fn=fn.drop(['Employee No'],axis=1)


# In[90]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[91]:


fn['Utilization%']=fn['Utilization%'].astype(float)


# In[92]:


fn.info()


# In[93]:


fn[fn.select_dtypes(include=['object']).columns] = fn[fn.select_dtypes(include=['object']).columns].apply(le.fit_transform/)


# In[94]:


fn.head()


# In[124]:


fn['promotion'].value_counts()


# In[95]:


fn.corr()


# In[96]:


plt.figure(figsize=(15,15))
sns.heatmap(fn.corr(),annot=True,cmap='coolwarm')


# In[125]:


fn.info()


# In[98]:


from sklearn.model_selection import train_test_split


# In[99]:


y=fn['Current Status']


# In[100]:


x=fn.drop(['Current Status'],axis=1)


# In[126]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=555)


# In[102]:


x_train.shape


# In[103]:


y_train.shape


# In[104]:


x_test.shape


# In[105]:


y_test.shape


# In[106]:


x_train.head()


# In[107]:


y_train.head()


# In[108]:


# CHI2 Test


# In[114]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# In[118]:


chi_test_fn=SelectKBest(score_func=chi2,k="all")


# In[119]:


fitted_fn=chi_test_fn.fit(abs(x),y)


# In[120]:


fitted_fn.scores_


# In[121]:


list(fitted_fn.scores_)


# In[122]:


df1_fn=pd.DataFrame({'Feature':x.columns,'Importance':fitted_fn.scores_})


# In[123]:


df1_fn.sort_values('Importance',ascending=False)


# In[47]:


#Boruta


# In[49]:


from boruta import BorutaPy


# In[50]:


from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()


# In[51]:


fn1=x


# In[52]:


fn_x=np.array(x)
fn_y=np.array(y)


# In[53]:


boruta_feature_selector=BorutaPy(rf,max_iter=30,verbose=2,random_state=555)


# In[55]:


boruta_feature_selector.fit(fn_x,fn_y)


# In[56]:


boruta_feature_selector.support_


# In[59]:


df2_fn=pd.DataFrame({'Feature':fn1.columns,'Importance':boruta_feature_selector.support_})


# In[60]:


df2_fn.sort_values('Importance',ascending=False)


# In[127]:


# model buliding(Logistic)


# In[128]:


from sklearn.linear_model import LogisticRegression
logreg_fn=LogisticRegression()


# In[129]:


logreg_fn.fit(x_train,y_train)


# In[130]:


pred_fn=logreg_fn.predict(x_test)
pred_fn


# In[131]:


from sklearn.metrics import confusion_matrix


# In[132]:


tab_fn=confusion_matrix(pred_fn,y_test)
tab_fn


# In[133]:


from sklearn.metrics import precision_score


# In[134]:


precision_score(y_test,pred_fn)


# In[138]:


tab_fn.diagonal().sum()*100/tab_fn.sum()


# In[135]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve


# In[136]:


log_roc_auc_fn=roc_auc_score(y_test,pred_fn)
log_roc_auc_fn    


# In[137]:


pred_value_prob_fn=logreg_fn.predict_proba(x_test)
fpr,tpr,threshold=roc_curve(y_test,pred_value_prob_fn[:,1])
plt.figure(figsize=(8,8))
plt.plot(fpr,tpr)
plt.xlabel("Fpr",size=20)
plt.ylabel("Tpr",size=20)
plt.title("AUROC Curve",size=20)
plt.grid()


# In[139]:


# Decision Tree


# In[ ]:




