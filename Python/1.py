import sys

def is_prime(n):
   for i in range(2, n):
      if n % i == 0:
         return False
   return True

def generate_twins(start, end):
   for i in range(start, end):
      j = i + 2
      if(is_prime(i) and is_prime(j)):
         f = open("myFirstFile.txt", "a")
         print("{:d} and {:d}".format(i, j) ,file=f)
         f.close()

x = int (input ("enter the no of digits : "))
start =(pow (10 , (x-1)) )+ 1
end = (pow (10 ,x) ) - 1 
generate_twins(start, end )

