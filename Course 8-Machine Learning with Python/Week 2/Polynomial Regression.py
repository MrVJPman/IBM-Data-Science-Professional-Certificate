import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
#%matplotlib inline

#!wget -O FuelConsumption.csv https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-Coursera/labs/Data_files/FuelConsumptionCo2.csv

df = pd.read_csv("FuelConsumption.csv")
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]


 
#============================Polynomial regression===========================
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])


poly = PolynomialFeatures(degree=2) #Generates 3 features, 1, x, x^2
train_x_poly = poly.fit_transform(train_x)

#Linear transform
clf = linear_model.LinearRegression()
train_y_ = clf.fit(train_x_poly, train_y)
# The coefficients
print ('Coefficients: ', clf.coef_) #Coefficients:  [[ 0.         47.85960214 -1.13036142]]
print ('Intercept: ',clf.intercept_) #Intercept:  [111.22123778]

#plotting the graph
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
XX = np.arange(0.0, 10.0, 0.1)
yy = clf.intercept_[0]+ clf.coef_[0][1]*XX+ clf.coef_[0][2]*np.power(XX, 2)
plt.plot(XX, yy, '-r' ) #non-linear red line
plt.xlabel("Engine size")
plt.ylabel("Emission")

#evaluating the model 
from sklearn.metrics import r2_score
test_x_poly = poly.fit_transform(test_x)
test_y_ = clf.predict(test_x_poly)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_ , test_y) )

#Try to use a polynomial regression with the dataset but this time with degree three (cubic).
#Does it result in better accuracy?

cubic_polynomial_model = PolynomialFeatures(degree=3)
train_x_cubic_polynomial = cubic_polynomial_model.fit_transform(train_x)
new_linear_model = linear_model.LinearRegression()
train_y_cubic_polynomial = new_linear_model.fit(train_x_cubic_polynomial, train_y)
test_x_cubic_polynomial = cubic_polynomial_model.fit_transform(test_x)
test_y_cubic_polynomial = new_linear_model.predict(test_x_cubic_polynomial)


# The coefficients
print ('Coefficients: ', new_linear_model.coef_)
print ('Intercept: ',new_linear_model.intercept_)
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
XX = np.arange(0.0, 10.0, 0.1)
yy = new_linear_model.intercept_[0]+ new_linear_model.coef_[0][1]*XX + new_linear_model.coef_[0][2]*np.power(XX, 2) + new_linear_model.coef_[0][3]*np.power(XX, 3)
plt.plot(XX, yy, '-r' )
plt.xlabel("Engine size")
plt.ylabel("Emission")
print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_cubic_polynomial - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_cubic_polynomial - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_cubic_polynomial , test_y) )
