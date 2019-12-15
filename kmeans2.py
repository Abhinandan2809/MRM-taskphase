#!/usr/bin/env python
# coding: utf-8

# In[2]:


from scipy.io import loadmat
mat =loadmat('/home/abhinandan/Desktop/ex7/bird_small.mat')
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import distance_matrix
mat


# In[3]:


x=np.array(mat['A'], np.float32)
xprime=x
plt.imshow(x)


# In[4]:


x=x.reshape(16384,3)


# In[5]:


plt.scatter(x[:,0],x[:,1],color="darkgreen",marker="o")


# In[6]:


clusters=x[np.random.choice(16384,16,replace=False)]
assignment=np.zeros((16384,1))


# In[7]:


def euclidean(x,clusters):
 diff=x-clusters
 diff=diff**2
 distance=diff.sum(axis=1) 
 return distance

    


# In[8]:


distance=np.zeros((16384,1))


# In[9]:


distance.shape


# In[10]:


for k in range(16):
    add=euclidean(x,clusters[k]).reshape(16384,1)
    distance=np.append(distance,add,axis=1)


# In[11]:


distance=np.delete(distance,0,1)


# In[12]:


assignment=np.argmin(distance,axis=1).reshape(16384,1)


# In[13]:


x=np.append(x,assignment,axis=1)
kernel=np.zeros((16,3))


# In[14]:


for k in range(16):
 assigned=x[x[:,3]==k]
 assigned=np.delete(assigned,3,1)
 kernel[k]=np.mean(assigned,axis=0)


# In[15]:


plt.scatter(x[:,0],x[:,1],color="darkgreen",marker="o")
plt.scatter(kernel[:,0],kernel[:,1],color="black",marker="x")


# In[16]:


x=np.delete(x,3,1)


# In[17]:


for i in range(16384):
    for k in range(16):
        if assignment[i]==k:
            x[i]=kernel[k]


# In[18]:


plt.imshow(x.reshape(128,128,3))


# In[ ]:





# In[ ]:




