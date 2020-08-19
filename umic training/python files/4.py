import numpy as np
from numpy.linalg import inv

# = np.array([1,2,3], dtype='int32')
y= np.random.randint(1,5,(20,1), dtype ='int32')

#print(y)
x = np.random.normal(0,1.0 , (20,20))  #becoz normalized
#print(x)



#print(k)
xinv = inv(x)
#print(k)
#print(kinv)

xtrans = x.transpose()
#print(ktrans)

#xinv = inv(x)
#print(xinv)
#xtrans = x.transpose()
#print(x)
a=np.matmul(xtrans,x)
b=np.matmul(xtrans,y)
j=inv(a)
theta = np.matmul(j,b)
print(theta)

#(xtrans)
