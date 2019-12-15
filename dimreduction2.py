#!/usr/bin/env python
# coding: utf-8

# In[99]:


from scipy.io import loadmat
mat =loadmat('/home/abhinandan/Desktop/ex7/ex7faces.mat')
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import distance_matrix
from scipy.linalg import svd
from sklearn.preprocessing import StandardScaler


# In[100]:


x=np.array(mat['X'], np.float32)
x.shape


# In[101]:


scaler=StandardScaler()
scaler.fit(x)
x=scaler.transform(x)


# In[102]:


plt.scatter(x[:,0],x[:,1],color="blue",marker="o")


# In[103]:


sigma=np.dot(x.T,x)/5000
k=100


# In[104]:


u,s,v=svd(sigma)
z=np.zeros((5000,k))


# In[105]:


ureduce=u[:,0:100]


# In[106]:


for i in range(1024):
 z[i]=np.dot(ureduce.T,x[i])


# In[ ]:





# In[107]:


xapprox=np.zeros((5000,1024))
for i in range(5000):
    xapprox[i]=np.dot(ureduce,z[i])


# In[108]:


plt.scatter(x[:,0],x[:,1],color="blue",marker="o")
plt.plot(xapprox[:,0],xapprox[:,1],"ro")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




