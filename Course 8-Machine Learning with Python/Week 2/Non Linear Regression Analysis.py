import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

#Linear, Quadratic, Cubic Functions
x = np.arange(-5.0, 5.0, 0.1)
y = 2*(x) + 3 
#y = np.power(x,2)
#y = 1*(x**3) + 1*(x**2) + 1*x + 3
y_noise = 2 * np.random.normal(size=x.size) 
#y_noise = 20 * np.random.normal(size=x.size)
ydata = y + y_noise
plt.plot(x, ydata,  'bo')
plt.plot(x,y, 'r') 
plt.ylabel('Dependent Variable')
plt.xlabel('Indepdendent Variable')
plt.show()



#Exponential and Logarithmic, Sigmoidal/Logistic
X = np.arange(-5.0, 5.0, 0.1)

Y= np.exp(X) #Y = np.log(X) #Y = 1-4/(1+np.power(3, X-2))

plt.plot(X,Y) 
plt.ylabel('Dependent Variable')
plt.xlabel('Indepdendent Variable')
plt.show()



#======================Non-Linear Regression example=================

import numpy as np
import pandas as pd

#downloading dataset
#!wget -nv -O china_gdp.csv https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-Coursera/labs/Data_files/china_gdp.csv
    
df = pd.read_csv("china_gdp.csv")
plt.figure(figsize=(8,5))
x_data, y_data = (df["Year"].values, df["Value"].values)
plt.plot(x_data, y_data, 'ro')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.show()

#====

X = np.arange(-5.0, 5.0, 0.1)
Y = 1.0 / (1.0 + np.exp(-X))

plt.plot(X,Y) 
plt.ylabel('Dependent Variable')
plt.xlabel('Indepdendent Variable')
plt.show()

def sigmoid(x, Beta_1, Beta_2):
    y = 1 / (1 + np.exp(-Beta_1*(x-Beta_2)))
    return y

beta_1 = 0.10
beta_2 = 1990.0

#logistic function
Y_pred = sigmoid(x_data, beta_1 , beta_2)

#plot initial prediction against datapoints
plt.plot(x_data, Y_pred*15000000000000.)
plt.plot(x_data, y_data, 'ro')

# Lets normalize our data
xdata =x_data/max(x_data)
ydata =y_data/max(y_data)

from scipy.optimize import curve_fit
popt, pcov = curve_fit(sigmoid, xdata, ydata)
#print the final parameters
print(" beta_1 = %f, beta_2 = %f" % (popt[0], popt[1]))

x = np.linspace(1960, 2015, 55)
x = x/max(x)
plt.figure(figsize=(8,5))
y = sigmoid(x, *popt)
plt.plot(xdata, ydata, 'ro', label='data')
plt.plot(x,y, linewidth=3.0, label='fit')
plt.legend(loc='best')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.show()



#Practise: what is the accuracy of our model?

#Creates train and test data
mask = np.random.rand(len(df)) < 0.8
x_train_data = xdata[mask]
x_test_data = xdata[~mask]
y_train_data = ydata[mask]
y_test_data = ydata[~mask]

#Creates the model using curve_fit
popt, pcov = curve_fit(sigmoid, x_train_data, y_train_data)

#Tests the model by creating predicted data
y_predicted_data = sigmoid(x_test_data, *popt)

#Regression Analysis : Checks for accuracy by comparing predicted data against test data
print("Mean absolute error: %.2f" % np.mean(np.absolute(y_predicted_data - y_test_data)))
print("Residual sum of squares (MSE): %.2f" % np.mean((y_predicted_data - y_test_data) ** 2))
print("R2-score: %.2f" % r2_score(y_predicted_data, y_test_data))
