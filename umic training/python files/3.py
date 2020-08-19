from math import sqrt

class Complex(object):

    def __init__(self, real, imag=0.0):
        self.real = real
        self.imag = imag

    def add (self, other):
        return Complex(self.real + other.real,
                       self.imag + other.imag) 

    def subtract (self, other):
        return Complex(self.real - other.real,
                       self.imag - other.imag)

    def multiply (self, other):
        return Complex(self.real*other.real - self.imag*other.imag,
                       self.imag*other.real + self.real*other.imag)
                      

    #def __div__(self, other):
    #    sr, si, or, oi = self.real, self.imag, \
    #                     other.real, other.imag # short forms
    #    r = float(or**2 + oi**2)
    #    return Complex((sr*or+si*oi)/r, (si*or-sr*oi)/r)
    def divide(self, other):

        r1 = self.real * other.real
        r2 = self.imag * other.imag
        denom = other.real**2 + other.imag**2
        ex1 = float((r1 + r2) / denom)
        i1 = self.real * other.imag * (-1)
        i2 = self.imag * other.real
        ex2 = float((i1 + i2) / denom)
        c = Complex(ex1, ex2)
        return c

    def inverse(self):
        return Complex(1,0).divide(self)

    def modulus(self):
        print(sqrt(self.real**2 + self.imag**2))

    def negative(self):   # defines -c (c is Complex)
        return Complex(-self.real, -self.imag)

    def eq(self, other):
        return self.real == other.real and self.imag == other.imag

    def ne(self, other):
        return not self.__eq__(other)

    def str(self):
        return '(%g, %g)' % (self.real, self.imag)

    def repr(self):
        return 'Complex' + str(self)

    #def pow(self, power):
    #    raise NotImplementedError\ 
    #          ('self**power is not yet impl. for Complex')

    def conjugate(self):
        return Complex(self.real, -self.imag)

    
    def display(self):
        print(self.real , "+" ,self.imag,"i")
        



a = Complex(2,1)
#a.conjugate().display()
a.inverse().display()
#display .repr(a)

#repr(a).displa)
b = Complex(2,-3)
c = b.divide(a)
#c.display()    
   


'''
         


def __init__(self, re, im):

    self.re = deepcopy(re)
    self.im = deepcopy(im)

def __str__(self):

    r1 = self.re
    i1 = self.im
    if(r1 > 0 and i1 > 0):
        r1 = str(r1)
        r1 +='+'
        if(abs(i1) != 1):
            i1 = str(i1)
            i1 += 'i'
        else:
            i1 = 'i'
    elif(r1 == 0 and i1 == 0):
        return '0'
    elif(r1 <= 0 and i1<0):
        if(r1 == 0):
            r1 = str(r1)
            r1 = ''
        if(i1 == -1):
            i1 = str(i1)
            i1 = '-i'
        else:
            i1 = str(i1)
            i1 += 'i'
    elif(r1 <= 0 and i1>0):
        if(r1 == 0):
            r1 = str(r1)
            r1 = ''
        else:
            r1 = str(r1)
            r1 += '+'
        if(i1 == 1):
            i1 = str(i1)
            i1 = 'i'
        else:
            i1 = str(i1)
            i1 += 'i'
    elif(r1 > 0 and i1 < 0):
        i1 = self.im
        i1 = str(i1)
        if(i1 != '-1'):
            i1 += 'i'
        else:
            i1 = '-i'
    if(i1 == 0):
        i1 = ''

    self.__repr__()
    ans = str(r1) + str(i1)
    return ans



def __add__(self, other):

    r1 = self.re + other.re
    i1 = self.im + other.im
    ans = Complex(r1,i1)
    return ans

def __sub__(self, other):
    r1 = self.re - other.re
    i1 = self.im - other.im
    ans = Complex(r1,i1)
    return ans

def __mul__(self, other):

    r1 = self.re * other.re
    r2 = self.im * other.im
    ex1 = r1 - r2
    i1 = self.re * other.im
    i2 = self.im * other.re
    ex2 = i1 + i2
    c = Complex(ex1,ex2)
    return c

def __truediv__(self, other):

    r1 = self.re * other.re
    r2 = self.im * other.im
    denom = other.re**2 + other.im**2
    ex1 = int((r1 + r2) / denom)
    i1 = self.re * other.im * (-1)
    i2 = self.im * other.re
    ex2 = int((i1 + i2) / denom)
    c = Complex(ex1, ex2)
    return c

def __eq__(self,other):

    if(self.re==other.re and self.im==other.im):
        return True
    else:
        return False

def norm(self):

    r1 = self.re
    i1 = self.im
    p1 = r1*r1
    p2 = i1*i1
    c = p1 + p2
    ans = int(sqrt(c))
    return ans  

def conj(self):
    return complex(self.re, - self.im)     

def cpow(c, n):
    r = Complex(1,0)
for i in range(n):
    r = r.__mul__(c)
return r

if name == 'main':
zero = Complex(0,0)
one = Complex(1,0)
iota = Complex(0,1)
minus_one = Complex(-1, 0)
minus_iota = Complex(0, -1)
c1 = Complex(1,1)
v = Complex(0,-1)
x = Complex(2, 3)
y = Complex(4, 5)
z = x + y
print(z)
print(x-y)
print(x*y)
print(x/y)
print(iota)
'''