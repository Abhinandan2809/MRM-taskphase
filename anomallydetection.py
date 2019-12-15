#!/usr/bin/env python
# coding: utf-8

# In[192]:


from scipy.io import loadmat
mat =loadmat('/home/abhinandan/Desktop/ex8/ex8data1.mat')
import numpy as np
from matplotlib import pyplot as plt


# In[193]:


xtrain=np.array(mat['X'], np.float32)
xval=np.array(mat['Xval'], np.float32)
yval=np.array(mat['yval'], np.float32)
xtrain.shape


# In[ ]:





# In[194]:


plt.scatter(xtrain[:,0],xtrain[:,1],color="blue",marker="o")


# In[195]:


mean=np.mean(xtrain,axis=0)
mean=mean.reshape(2,1)
variance=np.var(xtrain,axis=0).reshape(2,1)


# In[196]:


def gaussian(x,mean,variance):
 f = np.exp(-np.square(x-mean)/2*variance)/(np.sqrt(2*np.pi*variance))
 return f


# In[197]:


px=gaussian(xtrain[:,0].reshape(307,1),mean[0].reshape(1,1),variance[0].reshape(1,1))*gaussian(xtrain[:,1].reshape(307,1),mean[1].reshape(1,1),variance[1].reshape(1,1))


# In[198]:


def selectThreshold(yval, xval):
    """
    Find the best threshold (epsilon) to use for selecting outliers
    """
    best_epi = 0
    best_F1 = 0
    
    stepsize = (max(xval) -min(xval))/1000
    epi_range = np.arange(xval.min(),xval.max(),stepsize)
    for epi in epi_range:
        predictions = (xval<epi)[:,np.newaxis]
        tp = np.sum(predictions[yval==1]==1)
        fp = np.sum(predictions[yval==0]==1)
        fn = np.sum(predictions[yval==1]==0)
        
        # compute precision, recall and F1
        prec = tp/(tp+fp)
        rec = tp/(tp+fn)
        
        F1 = (2*prec*rec)/(prec+rec)
        
        if F1 > best_F1:
            best_F1 =F1
            best_epi = epi
        
    return best_epi, best_F1


# In[199]:


epsilon,F1=selectThreshold(yval,px)


# In[200]:


xtrain=np.append(xtrain,np.zeros((307,1)),axis=1)


# In[201]:


for i in range(307):
    if px[i]<=epsilon:
        xtrain[i][2]=1


# In[202]:


positive=xtrain[xtrain[:,2]==1]
negative=xtrain[xtrain[:,2]==0]
positive


# In[203]:


plt.scatter(positive[:,0],positive[:,1],color="red",marker="x")
plt.scatter(negative[:,0],negative[:,1],color="blue",marker="o")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




