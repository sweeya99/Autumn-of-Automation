import numpy as np
from numpy.linalg import inv

#random y of type int32
y= np.random.randint(1,5,(20,1), dtype ='int32')

#print(y)
x = np.random.normal(0,1.0 , (20,20))  #we can change values mean,std deviation
#print(x)

xinv = inv(x)
xtrans = x.transpose()
#print(xtrans)
#print(xinv)

a=np.matmul(xtrans,x)
b=np.matmul(xtrans,y)
j=inv(a)
theta = np.matmul(j,b)
print(theta)


