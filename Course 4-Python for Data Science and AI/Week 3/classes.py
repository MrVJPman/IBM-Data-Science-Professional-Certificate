import matplotlib.pyplot as plt
#%matplotlib inline 

# Create a class Circle

class Circle(object):
    
    # Constructor
    def __init__(self, radius=3, color='blue'):
        self.radius = radius
        self.color = color 
    
    # Method
    def add_radius(self, r):
        self.radius = self.radius + r
        return(self.radius)
    
    # Method
    def drawCircle(self):
        plt.gca().add_patch(plt.Circle((0, 0), radius=self.radius, fc=self.color))
        plt.axis('scaled')
        plt.show()  

# Create an object RedCircle

RedCircle = Circle(10, 'red')

# Find out the methods can be used on the object RedCircle

print(dir(RedCircle))

# Print the object attribute radius

print(RedCircle.radius)

# Print the object attribute color

print(RedCircle.color)

# Set the object attribute radius

RedCircle.radius = 1
RedCircle.radius

# Call the method drawCircle

RedCircle.drawCircle()

# Use method to change the object attribute radius

print('Radius of object:',RedCircle.radius)
RedCircle.add_radius(2)
print('Radius of object of after applying the method add_radius(2):',RedCircle.radius)
RedCircle.add_radius(5)
print('Radius of object of after applying the method add_radius(5):',RedCircle.radius)

# Create a blue circle with a given radius

BlueCircle = Circle(radius=100)

# Print the object attribute radius

BlueCircle.radius

# Print the object attribute color

BlueCircle.color
# Call the method drawCircle

BlueCircle.drawCircle()

# Create a new Rectangle class for creating a rectangle object

class Rectangle(object):
    
    # Constructor
    def __init__(self, width=2, height=3, color='r'):
        self.height = height 
        self.width = width
        self.color = color
    
    # Method
    def drawRectangle(self):
        plt.gca().add_patch(plt.Rectangle((0, 0), self.width, self.height ,fc=self.color))
        plt.axis('scaled')
        plt.show()

# Create a new object rectangle

SkinnyBlueRectangle = Rectangle(2, 10, 'blue')
# Use the drawRectangle method to draw the shape

SkinnyBlueRectangle.drawRectangle()

# Create a new object rectangle

FatYellowRectangle = Rectangle(20, 5, 'yellow')
# Use the drawRectangle method to draw the shape

FatYellowRectangle.drawRectangle()