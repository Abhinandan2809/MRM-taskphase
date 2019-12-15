#!/usr/bin/env python
# coding: utf-8

# In[91]:


from scipy.io import loadmat
mat =loadmat('/home/abhinandan/Desktop/ex7/ex7data2.mat')
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import distance_matrix


# In[92]:


x=np.array(mat['X'], np.float32)


# In[93]:


x=(x-x.mean())/x.std()


# In[94]:


plt.scatter(x[:,0],x[:,1],color="darkgreen",marker="o")


# In[95]:


clusters=x[np.random.choice(300,3,replace=False)]
assignment=np.zeros((300,1))


# In[96]:


def euclidean(x,clusters):
 diff=x-clusters
 diff=diff**2
 distance=diff.sum(axis=1) 
 return distance

    


# In[97]:


distance=np.zeros((300,1))


# In[98]:


distance.shape


# In[99]:


for k in range(3):
    add=euclidean(x,clusters[k]).reshape(300,1)
    distance=np.append(distance,add,axis=1)


# In[100]:


distance=np.delete(distance,0,1)


# In[101]:


assignment=np.argmin(distance,axis=1).reshape(300,1)


# In[102]:


x=np.append(x,assignment,axis=1)


# In[103]:


kernel1=x[x[:,2]==0]
kernel2=x[x[:,2]==1]
kernel3=x[x[:,2]==2]
kernel1=np.delete(kernel1,2,1)
kernel2=np.delete(kernel2,2,1)
kernel3=np.delete(kernel3,2,1)


# In[104]:


kernel1=np.mean(kernel1,axis=0)
kernel2=np.mean(kernel2,axis=0)
kernel3=np.mean(kernel3,axis=0)


# In[105]:


plt.scatter(x[:,0],x[:,1],color="darkgreen",marker="o")
plt.plot(kernel1[0],kernel1[1],color="black",marker="x")
plt.plot(kernel2[0],kernel2[1],color="black",marker="x")
plt.plot(kernel3[0],kernel3[1],color="black",marker="x")



# In[ ]:




