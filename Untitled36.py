#!/usr/bin/env python
# coding: utf-8

# In[67]:


from scipy.io import loadmat
mat =loadmat('/home/abhinandan/Desktop/ex5/ex5data1.mat')
import numpy as np
from matplotlib import pyplot as plt
mat


# In[68]:


xtrain=np.array(mat['X'], np.float32)
xprime=xtrain
ytrain=np.array(mat['y'], np.float32)
xcv=np.array(mat['Xval'], np.float32)
xcvsample=xcv
ycv=np.array(mat['yval'], np.float32)
plt.plot(xtrain,ytrain,"bo")
xcv.shape


# In[ ]:





# In[69]:


p=8
xcv.shape


# In[70]:


for i in range(2,p+1):
    xadd=xcvsample**i
    xcv=np.append(xcv,xadd,axis=1)
    
for i in range(2,p+1):
    xadd=xprime**i
    xtrain=np.append(xtrain,xadd,axis=1)
xtrain=((xtrain-xtrain.mean())/(xtrain.std()))
xcv=((xcv-xcv.mean())/(xcv.std()))
ycv=((ycv-ycv.mean())/(ycv.std()))
xcv=np.append(np.ones((21,1)),xcv,axis=1)


addrow=np.ones((12,1))
xtrain=np.append(addrow,xtrain,axis=1)
ytrain=((ytrain-ytrain.mean())/(ytrain.std()))
xsample=xtrain
ysample=ytrain    


# In[ ]:





# In[71]:


jtrain=[]
jcv=[]
count=[]
for i in range(1,(len(xtrain)+1)):
 xtrain=xsample[0:i,:]
 ytrain=ysample[0:i,:]
 param=np.zeros((p+1,1))
 for j in range(100000):
  hypo=np.dot(xtrain,param)
  gradient=(np.dot(xtrain.T,(hypo-ytrain))*0.00001/(i))
  param=param-gradient
 y_predicted=np.dot(xtrain,param)
 squared=(y_predicted-ytrain)**2
 squaredparam=param**2
 MSE=squared.mean()
 print(MSE)
 jtrain.append(MSE)
 y_predictcv=np.dot(xcv,param)
 squaredcv=(ycv-y_predictcv)**2
 MSECV=squaredcv.mean()
 print(MSECV)
 jcv.append(MSECV)
 count.append(i)
 



# In[72]:


plt.plot(count,jtrain)


# In[73]:


plt.plot(count,jcv)


# In[ ]:





# In[ ]:





# In[ ]:




