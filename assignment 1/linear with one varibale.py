
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd





data=pd.read_csv("/home/abhinandan/Desktop/machine-learning-ex1/ex1/ex1data1.csv", delimiter=',')




data





x=np.array([data['PL']], np.float32).reshape(97,1)




x





x.mean()





x=((x-x.mean())/(x.std()))




addrow=np.ones((97,1))
x=np.append(addrow,x,axis=1)





x





param=np.zeros((2,1))





y=np.array([data['PT']]).reshape(97,1)





y=((y-y.mean())/y.std())





n=list()
m=list()





for i in range(1500):
    
    hypo=np.dot(x,param)
    param=param-(np.dot(x.T,(hypo-y))*0.01/97)
    difference=(hypo-y)**2
    J=difference.mean()/2
    n.append(J)
    m.append(i)
    
print(param)    
    





predicted=(np.dot(x,param))
print(predicted)











k=(predicted-y)





k=k**2





k.mean()





plt.plot(m,n,"o")
plt.xlabel("epoch")
plt.ylabel("Cost function")







