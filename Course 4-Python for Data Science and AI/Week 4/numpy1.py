# Import the libraries

import time 
import sys
import numpy as np 

import matplotlib.pyplot as plt
#Code from Jupyter Notebook
#%matplotlib inline  

# Plotting functions
def Plotvec1(u, z, v):
    
    ax = plt.axes()
    ax.arrow(0, 0, *u, head_width=0.05, color='r', head_length=0.1)
    plt.text(*(u + 0.1), 'u')
    
    ax.arrow(0, 0, *v, head_width=0.05, color='b', head_length=0.1)
    plt.text(*(v + 0.1), 'v')
    ax.arrow(0, 0, *z, head_width=0.05, head_length=0.1)
    plt.text(*(z + 0.1), 'z')
    plt.ylim(-2, 2)
    plt.xlim(-2, 2)

def Plotvec2(a,b):
    ax = plt.axes()
    ax.arrow(0, 0, *a, head_width=0.05, color ='r', head_length=0.1)
    plt.text(*(a + 0.1), 'a')
    ax.arrow(0, 0, *b, head_width=0.05, color ='b', head_length=0.1)
    plt.text(*(b + 0.1), 'b')
    plt.ylim(-2, 2)
    plt.xlim(-2, 2)
    
# Create a python list
a = ["0", 1, "two", "3", 4]

# Print each element
print("a[0]:", a[0])
print("a[1]:", a[1])
print("a[2]:", a[2])
print("a[3]:", a[3])
print("a[4]:", a[4])

# import numpy library
import numpy as np 

# Create a numpy array
a = np.array([0, 1, 2, 3, 4])

# Print each element
print("a[0]:", a[0])
print("a[1]:", a[1])
print("a[2]:", a[2])
print("a[3]:", a[3])
print("a[4]:", a[4])

# Check the type of the array
type(a)

# Check the type of the values stored in numpy array
a.dtype

# Create a numpy array
b = np.array([3.1, 11.02, 6.2, 213.2, 5.2])

# Check the type of array
type(b)

# Check the value type
b.dtype

# Create numpy array
c = np.array([20, 1, 2, 3, 4])

# Assign the first element to 100
c[0] = 100

# Slicing the numpy array
d = c[1:4]

# Set the fourth element and fifth element to 300 and 400
c[3:5] = 300, 400

# Create the index list
select = [0, 2, 3]

# Use List to select elements
d = c[select]

# Assign the specified elements to new value
c[select] = 100000

# Create a numpy array
a = np.array([0, 1, 2, 3, 4])

# Get the size of numpy array
a.size

# Get the number of dimensions of numpy array
a.ndim

# Get the shape/size of numpy array
a.shape

# Create a numpy array
a = np.array([1, -1, 1, -1])

# Get the mean of numpy array
mean = a.mean()

# Get the standard deviation of numpy array
standard_deviation=a.std()

# Create a numpy array
b = np.array([-1, 2, 3, 4, 5])

# Get the biggest value in the numpy array
max_b = b.max()

# Get the smallest value in the numpy array
min_b = b.min()

u = np.array([1, 0])
u
v = np.array([0, 1])
v

# Numpy Array Addition
z = u + v
z #array([1, 1])

# Plot numpy arrays
Plotvec1(u, z, v)

# Create a numpy array
y = np.array([1, 2])
# Numpy Array Multiplication
z = 2 * y 
z #array([2, 4])

# Create a numpy array
u = np.array([1, 2])
u

# Create a numpy array
v = np.array([3, 2])
v

# Calculate the production of two numpy arrays
z = u * v
z #array([3, 4])

# Calculate the dot product
np.dot(u, v)

# Create a constant to numpy array
u = np.array([1, 2, 3, -1]) 

# Add the constant to array
u + 1 #array([2, 3, 4, 0])

# The value of pie
np.pi

# Create the numpy array in radians
x = np.array([0, np.pi/2 , np.pi])

# Calculate the sin of each elements
y = np.sin(x)
y #array([0.0000000e+00, 1.0000000e+00, 1.2246468e-16])

# Makeup a numpy array within [-2, 2] and 5 elements
np.linspace(-2, 2, num=5) #array([-2., -1.,  0.,  1.,  2.])

# Makeup a numpy array within [-2, 2] and 9 elements
np.linspace(-2, 2, num=9) #array([-2. , -1.5, -1. , -0.5,  0. ,  0.5,  1. ,  1.5,  2. ])

# Makeup a numpy array within [0, 2*pi] and 100 elements 
x = np.linspace(0, 2*np.pi, num=100)

# Calculate the sine of x list and  Plot the result
y = np.sin(x)
plt.plot(x, y)




#Implement the following vector subtraction in numpy: u-v
u = np.array([1, 0])
v = np.array([0, 1])
w = u-v

#Multiply the numpy array z with -2:
z = np.array([2, 4])
z_new = z * -2

#Consider the list [1, 2, 3, 4, 5] and [1, 0, 1, 0, 1], and cast both lists to a numpy array then multiply them together:
a = np.array([1, 2, 3, 4, 5])
b = np.array([1, 0, 1, 0, 1])
product = a * b

#Convert the list [-1, 1] and [1, 1] to numpy arrays a and b. 
#Then, plot the arrays as vectors using the fuction Plotvec2 and find the dot product:
a = np.array([-1,1])
b = np.array([1,1])
Plotvec2(a,b)
dot_product = np.dot(a, b)

#Convert the list [1, 0] and [0, 1] to numpy arrays a and b. 
#Then, plot the arrays as vectors using the function Plotvec2 and find the dot product:
a = np.array([1,0])
b = np.array([0,1])
Plotvec2(a,b)
dot_product = np.dot(a, b)

#Convert the list [1, 1] and [0, 1] to numpy arrays a and b. 
#Then plot the arrays as vectors using the fuction Plotvec2 and find the dot product:
a = np.array([1,1])
b = np.array([0,1])
Plotvec2(a,b)
dot_product = np.dot(a, b)

#Why are the results of the dot product for [-1, 1] and [1, 1] and the dot product for [1, 0] and [0, 1] zero, 
#but not zero for the dot product for [1, 1] and [0, 1]?

#Answer : Because the first two are perpendicular to each other 