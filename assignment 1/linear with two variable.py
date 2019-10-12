


import numpy as np
from matplotlib import pyplot as plt





import pandas as pd
data=pd.read_csv('/home/abhinandan/Desktop/ex1data2.csv', delimiter=',')




print(data)





addrow=np.ones((1,47))
print(addrow)






x=np.array([[data['A']], [data['B']]], dtype=np.float)
x=x.reshape(2,47)
x=((x-x.mean())/x.std())
x=np.vstack((addrow,x))





print(x)
x.shape





xs=x.T
xs=xs.reshape(47,3)
xs.shape











print(xs)





param=np.zeros((3,1))
















ys=np.array([data['C']]).T
print(ys)
print(ys.shape)
print(param.shape)
hypo=np.zeros((47,1))



ys=ys.reshape(47,1)
ys=(ys-ys.mean())/(ys.std())
n=list()
m=list()
for i in range(1500):
    hypo=np.dot(xs, param)
    xsprime=xs.T
    ydiff=hypo-ys
    squared=ydiff**2
    summission=np.dot(xsprime,ydiff)
    summission=summission*0.01/47
    J=(squared.mean()/2)
    n.append(i)
    m.append(J)
    temp=param-summission
    param=temp
print(param)


plt.plot(n,m,"o")
plt.xlabel("epoch")
plt.ylabel("Cost function")
y_pred=np.dot(xs,param)





difference2=(y_pred-ys)**2
MSE=difference2.mean()
print(MSE)





