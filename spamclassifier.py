#!/usr/bin/env python
# coding: utf-8

# In[53]:


from scipy.io import loadmat
mat =loadmat('/home/abhinandan/Desktop/ex6/ex6data3.mat')
import numpy as np
from matplotlib import pyplot as plt
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize


# In[54]:


email=open("/home/abhinandan/Desktop/ex6/emailSample2.txt","rt").read()


# In[55]:


import re
def remove_html_tags(text):
    """Remove html tags from a string"""
    import re
    clean = re.compile('<.*?>')
    text=re.sub(clean, '', text)
    return text
def preprocess(email):
 email=email.lower().rstrip()  
 email=email.replace("$","dollar")
 punctuation=[",",".","?",";"":",">","<","'","(",")","-","_"]
 for i in punctuation:
  email=email.replace(i,"")
 email=remove_html_tags(email)
 email=email.strip()
 email= re.sub('\S+@\S+',"emailaddr", email) 
 email=re.sub("http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+","httpaddr",email)
 email = re.sub(r'[0-9]+',"number", email) 
 email=re.sub('\s+',' ',email)
 return email


# In[56]:


def stemming (email):
 prep=preprocess(email)
 ps = PorterStemmer()
 words=prep.split()
 for i in range(len(words)):
    words[i]=ps.stem(words[i])
 prep=" "
 prep=prep.join(words)
 prep=prep.replace(":","")
 return prep
 


# In[57]:


prep=stemming(email)
prep


# In[58]:


vocablist=open("/home/abhinandan/Desktop/ex6/vocab.txt","rt")


# In[59]:


vocab={}
for i in vocablist:
    i=i.split()
    key=i[1]
    vocab[key]=i[0]
    
    


# In[60]:


features=np.zeros((1899,1))


# In[61]:


for key in vocab.keys():
    if key in prep:
        features[int(vocab[key])]=1


# In[62]:


from scipy.io import loadmat
mat =loadmat('/home/abhinandan/Desktop/ex6/spamTrain.mat')
import numpy as np
from matplotlib import pyplot as plt
mat2 =loadmat('/home/abhinandan/Desktop/ex6/spamTest.mat')
mat2


# In[63]:


xtrain=np.array(mat['X'], np.float32)
ytrain=np.array(mat['y'], np.float32)


# In[64]:


xtest=np.array(mat2['Xtest'], np.float32)
ytest=np.array(mat2['ytest'], np.float32)


# In[65]:


from sklearn.svm import SVC
model=SVC(C=0.1,kernel="linear")
clf=model.fit(xtrain,ytrain.reshape(4000,))


# In[66]:


model.score(xtrain,ytrain)*100


# In[ ]:


model.score(xtest,ytest)*100


# In[ ]:




