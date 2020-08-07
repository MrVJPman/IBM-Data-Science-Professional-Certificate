#Model Development
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# path of data 
path = 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/automobileEDA.csv'
df = pd.read_csv(path)
#print(df.head())

from sklearn.linear_model import LinearRegression

lm = LinearRegression()
#print(lm)

#Fit the linear model using highway-mpg.
X = df[['highway-mpg']]
Y = df['price']
lm.fit(X,Y)
Yhat=lm.predict(X)
Yhat[0:5]   

lm.intercept_
lm.coef_

#Question #1 a):
#Create a linear regression object?
new_lm = LinearRegression()

#Question #1 b):
#Train the model using 'engine-size' as the independent variable and 'price' as the dependent variable?
X = df[['engine-size']]
Y = df['price']
new_lm.fit(X,Y)

#Question #1 c):
#Find the slope and intercept of the model?
new_lm.coef_ #slope
new_lm.intercept_ #intercept
 
#Question #1 d):
#What is the equation of the predicted line.
#Answer Price = 166 * Engine-Size - 7963







#Multiple Linear Regression
Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
lm.fit(Z, df['price'])
lm.intercept_
lm.coef_

#Question #2 a):
#Create and train a Multiple Linear Regression model "lm2" where the response variable is price, 
#and the predictor variable is 'normalized-losses' and 'highway-mpg'.
lm2 = LinearRegression()
predictor_variables = df[["normalized-losses", "highway-mpg"]]
response_variables = df["price"]
lm2.fit(predictor_variables, response_variables)

#Question #2 b):
#Find the coefficient of the model?
lm2.coef_


#2) Model Evaluation using Visualization

# import the visualization package: seaborn
import seaborn as sns
#%matplotlib inline 

width = 12
height = 10
plt.figure(figsize=(width, height))
sns.regplot(x="highway-mpg", y="price", data=df)
plt.ylim(0,)
#plt.show()


plt.figure(figsize=(width, height))
sns.regplot(x="peak-rpm", y="price", data=df)
plt.ylim(0,)
#plt.show()

#Question #3:
#Given the regression plots above is "peak-rpm" or "highway-mpg" more strongly correlated with "price". 
#highway rpm is more correlated
#Use the method ".corr()" to verify your answer.
df[["peak-rpm", "highway-mpg", "price"]].corr()









#Residual Plot : shows the residuals on the vertical y-axis and the independent variable on the horizontal x-axis.


#residual (e) : observed value (y) - predicted value (Yhat)
#When we look at a regression plot, the residual is the distance from the data point to the fitted regression line.

#So what is a residual plot?

width = 12
height = 10
plt.figure(figsize=(width, height))
sns.residplot(df['highway-mpg'], df['price'])
#plt.show()


Y_hat = lm.predict(Z)
plt.figure(figsize=(width, height))


plt.clf()
ax1 = sns.distplot(df['price'], hist=False, color="r", label="Actual Value")
sns.distplot(Yhat, hist=False, color="b", label="Fitted Values" , ax=ax1)


plt.title('Actual vs Fitted Values for Price')
plt.xlabel('Price (in dollars)')
plt.ylabel('Proportion of Cars')

plt.show()
plt.close()

#======================================================================


def PlotPolly(model, independent_variable, dependent_variabble, Name):
    x_new = np.linspace(15, 55, 100)
    y_new = model(x_new)

    plt.plot(independent_variable, dependent_variabble, '.', x_new, y_new, '-')
    plt.title('Polynomial Fit with Matplotlib for Price ~ Length')
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    fig = plt.gcf()
    plt.xlabel(Name)
    plt.ylabel('Price of Cars')

    plt.show()
    plt.close()
    
x = df['highway-mpg']
y = df['price']
# Here we use a polynomial of the 3rd order (cubic) 
f = np.polyfit(x, y, 3)
p = np.poly1d(f)
print(p)
PlotPolly(p, x, y, 'highway-mpg')
np.polyfit(x, y, 3)


#Question #4:
#Create 11 order polynomial model with the variables x and y from above?
f_new = np.polyfit(x, y, 11)
p_new = np.poly1d(f_new)
print(p_new)
PlotPolly(p_new, x, y, 'highway-mpg')    









from sklearn.preprocessing import PolynomialFeatures
pr=PolynomialFeatures(degree=2)
Z_pr=pr.fit_transform(Z)
Z.shape #Originally 201 x 4
Z_pr.shape #Originally 201 x 15 





#Pipeline : Perform multiple s teps
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
Input=[('scale',StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model',LinearRegression())]
pipe=Pipeline(Input)
pipe.fit(Z,y)
ypipe=pipe.predict(Z)
ypipe[0:4]


#Question #5:
#Create a pipeline that Standardizes the data, 
#then perform prediction using a linear regression model using the features Z and targets y
# Write your code below and press Shift+Enter to execute 
Input=[('scale',StandardScaler()), ('model',LinearRegression())] #initialize the 1st, 2nd pipelines
pipe=Pipeline(Input) #sets up the pipeline
pipe.fit(Z,y)
pipe=pipe.predict(Z)
ypipe[0:10]






#Part 4: Measures for In-Sample Evaluation

#Model 1: Simple Linear Regression
#highway_mpg_fit
lm.fit(X, Y)
# Find the R^2
print('The R-square is: ', lm.score(X, Y))
Yhat=lm.predict(X)
print('The output of the first four predicted value is: ', Yhat[0:4])


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df['price'], Yhat)
print('The mean square error of price and predicted value is: ', mse)

#Model 2: Multiple Linear Regression
# fit the model 
lm.fit(Z, df['price'])
# Find the R^2
print('The R-square is: ', lm.score(Z, df['price']))

Y_predict_multifit = lm.predict(Z)
print('The mean square error of price and predicted value using multifit is: ', \
      mean_squared_error(df['price'], Y_predict_multifit))

#Model 3: Polynomial Fit
from sklearn.metrics import r2_score
r_squared = r2_score(y, p(x))
print('The R-square value is: ', r_squared)

#MSE
mean_squared_error(df['price'], p(x))


#Part 5: Prediction and Decision Making

import matplotlib.pyplot as plt
import numpy as np

#%matplotlib inline 

#Create a new input
new_input=np.arange(1, 100, 1).reshape(-1, 1)
lm.fit(X, Y)

#Produce a prediction
yhat=lm.predict(new_input)
yhat[0:5]

#
plt.plot(new_input, yhat)
plt.show()


#What is a good R-squared value? R=1.0
#What is a good MSE? MSE=0
