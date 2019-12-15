#!/usr/bin/env python
# coding: utf-8

# In[195]:


from scipy.io import loadmat
mat =loadmat('/home/abhinandan/Desktop/ex5/ex5data1.mat')
import numpy as np
from matplotlib import pyplot as plt
mat


# In[196]:


xtrain=np.array(mat['X'], np.float32)
xprime=xtrain
ytrain=np.array(mat['y'], np.float32)
plt.plot(xtrain,ytrain,"bo")
xtrain=np.array(mat['X'], np.float32)
xcv=np.array(mat['Xval'], np.float32)
ycv=np.array(mat['yval'], np.float32)
xcv.shape



# In[197]:


xtrain.shape


# In[198]:


xtrain=((xtrain-xtrain.mean())/(xtrain.std()))
ytrain=((ytrain-ytrain.mean())/(ytrain.std()))
xcv=((xcv-xcv.mean())/(xcv.std()))
ycv=((ycv-ycv.mean())/(ycv.std()))
addrow=np.ones((12,1))
xtrain=np.append(addrow,xtrain,axis=1)
xcv=np.append(np.ones((21,1)),xcv,axis=1)
param=np.zeros((2,1))
xsample=xtrain
ysample=ytrain


# In[199]:


for i in range(10000):
    hypo=np.dot(xtrain,param)
    gradient=(np.dot(xtrain.T,(hypo-ytrain))*0.1/12)
    param=param-gradient
y_predicted=np.dot(xtrain,param)
squared=(y_predicted-ytrain)**2
MSE=squared.mean()
print(MSE)
plt.plot(xprime,y_predicted)
plt.plot(xprime,ytrain,"bo")


# In[200]:


jtrain=[]
jcv=[]
count=[]
for i in range(1,(len(xtrain)+1)):
 xtrain=xsample[0:i,:]
 ytrain=ysample[0:i,:]
 param=np.zeros((2,1))
 for j in range(10000):
  hypo=np.dot(xtrain,param)
  gradient=(np.dot(xtrain.T,(hypo-ytrain))*0.1/(i))
  param=param-gradient
 y_predicted=np.dot(xtrain,param)
 squared=(y_predicted-ytrain)**2
 MSE=squared.mean()
 print(MSE)
 jtrain.append(MSE)
 y_predictcv=np.dot(xcv,param)
 squaredcv=(ycv-y_predictcv)**2
 MSECV=squaredcv.mean()
 print(MSECV)
 jcv.append(MSECV)
 count.append(i)
 



# In[201]:


print(jtrain)
print(jcv)
print(count)


# In[ ]:





# In[ ]:





# In[209]:


plt.plot(count,jcv)


# In[212]:


plt.plot(count,jtrain)


# In[ ]:




