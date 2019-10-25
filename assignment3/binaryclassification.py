

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd





data=pd.read_csv("/home/abhinandan/Desktop/ex2data1.csv", delimiter=',')











x=np.array([data['1'],data['2']]).T.reshape(100,2)
x=((x-x.mean())/(x.std()))





y=np.array([data['3']]).T.reshape(100,1)
ones=np.ones((100,1))
x=np.append(ones,x,axis=1)
param=np.zeros((3,1))





def sigmoid(z):
    return 1/(1+np.exp(-z))



 
m=list()
n=list()





for i in range(50000):
    hyp=sigmoid(np.dot(x,param))    
    param=param-(np.dot(x.T,(hyp-y))*0.1/100)
    J1=y*np.log(hyp)
    J2=(1-y)*np.log(1-hyp)
    J=((J2+J1).mean())*(-1)
    n.append(J)
    m.append(i)
print(param)    
    





plt.plot(m,n,"red")





for i in range(len(hyp)):
    if hyp[i]>=0.5:
        hyp[i]=1
    else:
        hyp[i]=0
hyp.shape





difference=hyp-y
count=0
for i in range(len(difference)):
    if difference[i]==0:
        count+=1
accuracy=count/100   
print(accuracy*100)













