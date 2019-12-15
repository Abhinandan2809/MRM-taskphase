#!/usr/bin/env python
# coding: utf-8

# In[82]:


from scipy.io import loadmat
mat =loadmat('/home/abhinandan/Desktop/ex6/ex6data1.mat')
import numpy as np
from matplotlib import pyplot as plt
mat


# In[83]:


xtrain=np.array(mat['X'], np.float32)
xprime=xtrain
ytrain=np.array(mat['y'], np.float32)


# In[84]:


df=np.append(xtrain,ytrain,axis=1)


# In[85]:


positive=df[df[:,2]==1]
negative=df[df[:,2]==0]


# In[86]:


plt.scatter(positive[:,0],positive[:,1],color="red",marker="+")
plt.scatter(negative[:,0],negative[:,1],color="green",marker="o")


# In[87]:


from sklearn.model_selection import train_test_split


# In[88]:


x_train,x_test,y_train,y_test=train_test_split(xtrain,ytrain,test_size=0.2)


# In[89]:


from sklearn.svm import SVC
model=SVC(C=100,kernel="linear")


# In[90]:


clf=model.fit(x_train,y_train.reshape(40,))


# In[91]:


model.score(x_test,y_test)


# In[92]:


def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy


# In[93]:


def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


# In[94]:


fig, ax = plt.subplots()
X0, X1= df[:,0], df[:,1]
xx, yy=make_meshgrid(X0,X1)
plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
ax.scatter(X0, X1, c=df[:,2], cmap=plt.cm.coolwarm, s=20, edgecolors='k')
ax.set_xticks(())
ax.set_yticks(())
ax.legend()
plt.show()


# In[ ]:


]

