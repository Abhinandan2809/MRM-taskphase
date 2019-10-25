


import numpy as np
from matplotlib import pyplot as plt
import pandas as pd





data=pd.read_csv("/home/abhinandan/Desktop/ex2data2.csv", delimiter=',')










x=np.ones((118,1))
for i in range(7):
    for j in range(7-i):
     if i==0 and j==0:
        continue
     else:
        new1=np.array([data['1']**i]).reshape(118,1)
        new2=np.array([data['2']**j]).reshape(118,1)
        new=new1*new2
        x=np.append(x,new,axis=1)    
        











y=np.array([data['3']]).T.reshape(118,1)
param=np.zeros((28,1))





def sigmoid(z):
    return 1/(1+np.exp(-z))





m=list()
n=list()





for i in range(15000):
    hyp=sigmoid(np.dot(x,param))    
    param=param-((np.dot(x.T,(hyp-y))+(10*param))*0.1/118)
    J1=y*np.log(hyp)
    J2=(1-y)*np.log(1-hyp)
    J=((J2+J1).mean())*(-1)+(10*(param**2).mean()/2)*(-1)
    n.append(J)
    m.append(i)
print(param)    





plt.plot(m,n,"red")





for i in range(len(hyp)):
    if hyp[i]>=0.5:
        hyp[i]=1
    else:
        hyp[i]=0





difference=hyp-y
count=0
for i in range(len(difference)):
    if difference[i]==0:
        count+=1
accuracy=count/100   
print(accuracy*100)







