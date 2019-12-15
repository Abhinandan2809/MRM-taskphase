#!/usr/bin/env python
# coding: utf-8

# In[25]:


from scipy.io import loadmat
mat =loadmat('/home/abhinandan/Desktop/ex7/ex7data1.mat')
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import distance_matrix
from scipy.linalg import svd
from sklearn.preprocessing import StandardScaler


# In[26]:


x=np.array(mat['X'], np.float32)


# In[27]:


scaler=StandardScaler()
scaler.fit(x)
x=scaler.transform(x)


# In[28]:


plt.scatter(x[:,0],x[:,1],color="blue",marker="o")


# In[29]:


sigma=np.dot(x.T,x)/50


# In[30]:


u,s,v=svd(sigma)
z=np.zeros((50,1))


# In[31]:


ureduce=u[:,0].reshape(2,1)


# In[32]:


for i in range(50):
 z[i]=np.dot(ureduce.T,x[i])
z


# In[ ]:





# In[33]:


xapprox=np.zeros((50,2))
for i in range(50):
    xapprox[i]=np.dot(ureduce,z[i])


# In[34]:


plt.scatter(x[:,0],x[:,1],color="blue",marker="o")
plt.plot(xapprox[:,0],xapprox[:,1],"ro")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




