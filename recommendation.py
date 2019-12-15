#!/usr/bin/env python
# coding: utf-8

# In[29]:


from scipy.io import loadmat
mat =loadmat('/home/abhinandan/Desktop/ex8/ex8_movies.mat')
import numpy as np
from matplotlib import pyplot as plt
mat


# y=np.array(mat['Y'], np.float32)
# r=np.array(mat['R'], np.float32)
# r.shape
# 

# In[30]:


x=np.random.rand(1682,100)
theta=np.random.rand(943,100)
r.shape
theta.shape


# In[31]:


xgrad=np.zeros(x.shape)
thetagrad=np.zeros(theta.shape)


# In[32]:


for i in range(1682):
    idx=np.nonzero(r[i,:])
    theta_temp=theta[idx,:]
    y_temp=y[i,idx]
    x_grad=(((x[i,:].reshape(100,1).dot(theta_temp.T.reshape(100,452)))-y_temp).dot(theta_temp))
    


# In[ ]:





# In[ ]:


for i in range(943):
    idx=np.nonzero(r[i,:])
    x_temp=x[idx,:]
    y_temp=y[idx,i]
    theta_grad[i,:]=(((x_temp.dot(theta[i,:].T))-y_temp).T.dot(x_temp))
    


# In[ ]:


alpha=0.1
x=x-alpha*x_grad
theta=theta-alpha*theta_grad


# In[ ]:


y_predict=theta.T.dot(x)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




