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

    def conjugate(self):
        return Complex(self.real, -self.imag)

    
    def display(self):
        print(self.real , "+" ,self.imag,"i")
        

a = Complex(1,2)
#### a would refer to 1+2*i 
a .display()
#### prints out “1 + 2i”
b = Complex(2,-3)
c = b.add(a)
c.display()
#### prints out “3 - 1i”
c.conjugate().display() ### prints “3+1i”


