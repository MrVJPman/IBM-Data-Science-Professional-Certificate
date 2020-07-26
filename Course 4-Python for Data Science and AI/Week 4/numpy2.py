# Import the libraries
import numpy as np 
import matplotlib.pyplot as plt

# Create a list
a = [[11, 12, 13], [21, 22, 23], [31, 32, 33]]

# Convert list to Numpy Array
# Every element is the same type
A = np.array(a)

# Show the numpy array dimensions
A.ndim # 2

# Show the numpy array shape
A.shape #(3, 3)

# Show the numpy array size
A.size #9

# Access the element on the second row and third column
A[1, 2]

# Access the element on the second row and third column
A[1][2]

# Access the element on the first row and first column
A[0][0]

# Access the element on the first row and first and second columns
A[0][0:2]

# Access the element on the first and second rows and third column
A[0:2, 2]

# Create a numpy array X , Y
X = np.array([[1, 0], [0, 1]]) 
Y = np.array([[2, 1], [1, 2]]) 
Z = X + Y # array([[3, 1],[1, 3]])

# Create a numpy array Y
Y = np.array([[2, 1], [1, 2]]) 
Z = 2 * Y
Z #np.array([[4, 2], [2, 4]])

# Create a numpy array Y, X
Y = np.array([[2, 1], [1, 2]]) 
X = np.array([[1, 0], [0, 1]]) 
Z = X * Y #array([[2, 0], [0, 2]])

# Create a matrix A, B
A = np.array([[0, 1, 1], [1, 0, 1]])
B = np.array([[1, 1], [1, 1], [-1, 1]])

# Calculate the dot product
Z = np.dot(A,B) #array([[0, 2], [0, 2]])

# Calculate the sine of Z
np.sin(Z)       

# Create a matrix C and Get the transposed of C
C = np.array([[1,1],[2,2],[3,3]])
C.T
