#Model Evaluation and Refinement

import pandas as pd
import numpy as np

# Import clean data 
path = 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/module_5_auto.csv'
df = pd.read_csv(path)

df.to_csv('module_5_auto.csv')

df = df._get_numeric_data()
df.head()

#%%capture
#! pip install ipywidgets

from IPython.display import display
from IPython.html import widgets 
from IPython.display import display
from ipywidgets import interact, interactive, fixed, interact_manual

def DistributionPlot(RedFunction, BlueFunction, RedName, BlueName, Title):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))

    ax1 = sns.distplot(RedFunction, hist=False, color="r", label=RedName)
    ax2 = sns.distplot(BlueFunction, hist=False, color="b", label=BlueName, ax=ax1)

    plt.title(Title)
    plt.xlabel('Price (in dollars)')
    plt.ylabel('Proportion of Cars')

    plt.show()
    plt.close()
    
def PollyPlot(xtrain, xtest, y_train, y_test, lr,poly_transform):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))


    #training data 
    #testing data 
    # lr:  linear regression object 
    #poly_transform:  polynomial transformation object 

    xmax=max([xtrain.values.max(), xtest.values.max()])

    xmin=min([xtrain.values.min(), xtest.values.min()])

    x=np.arange(xmin, xmax, 0.1)


    plt.plot(xtrain, y_train, 'ro', label='Training Data')
    plt.plot(xtest, y_test, 'go', label='Test Data')
    plt.plot(x, lr.predict(poly_transform.fit_transform(x.reshape(-1, 1))), label='Predicted Function')
    plt.ylim([-10000, 60000])
    plt.ylabel('Price')
    plt.legend()

y_data = df["price"]
x_data = df.drop("price", axis=1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.15, random_state=1)
#percentage of the data used for testing would be 15%, 85% for training
print("number of test samples :", x_test.shape[0])
print("number of training samples:",x_train.shape[0])   



#Question #1):
#Use the function "train_test_split" to split up the data set such that 40%
#of the data samples will be utilized for testing, set the parameter "random_state"
#equal to zero. The output of the function should be the following: "x_train_1" , 
#"x_test_1", "y_train_1" and "y_test_1".

x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(x_data, y_data, test_size = 0.4, random_state = 0)
print("number of test samples :", x_test_1.shape[0])
print("number of training samples:",x_train_1.shape[0])


from sklearn.linear_model import LinearRegression
lre=LinearRegression()
#we fit the model using the feature horsepower
lre.fit(x_train[['horsepower']], y_train) #we always fit with training data
lre.score(x_test[['horsepower']], y_test) #R2 score greater
lre.score(x_train[['horsepower']], y_train) #R2 score lower

#Question #2):
#Find the R^2 on the test data using 90% of the data for training data

# Write your code below and press Shift+Enter to execute 
x_train1, x_test1, y_train1, y_test1 = train_test_split(x_data, y_data, test_size = 0.1, random_state = 0) 
#10% for testing
#90% for training
lre = LinearRegression()
lre.fit(x_train1[["horsepower"]], y_train1)
lre.score(x_test1[["horsepower"]], y_test1)













#Cross-validation Score
from sklearn.model_selection import cross_val_score
#We input the object, the feature in this case ' horsepower', the target data (y_data). 
#The parameter 'cv' determines the number of folds; in this case 4.
Rcross = cross_val_score(lre, x_data[["horsepower"]], y_data, cv=4)
print("The mean of the folds are", Rcross.mean(), "and the standard deviation is" , Rcross.std())

#We can use negative squared error as a score by setting the parameter 'scoring' metric to 'neg_mean_squared_error'.
-1 * cross_val_score(lre,x_data[['horsepower']], y_data,cv=4,scoring='neg_mean_squared_error')

#Question #3):
#Calculate the average R^2 using two folds,
#find the average R^2 for the second fold utilizing the horsepower as a feature :

# Write your code below and press Shift+Enter to execute 
Rcross_2 = cross_val_score(lre, x_data[["horsepower"]], y_data, cv=2)
Rcross_2.mean()

from sklearn.model_selection import cross_val_predict
#We input the object, the feature in this case 'horsepower' , 
#the target data y_data. 
#he parameter 'cv' determines the number of folds; in this case 4.
#We can produce an output:
yhat = cross_val_predict(lre,x_data[['horsepower']], y_data,cv=4)
yhat[0:5]









#Part 2: Overfitting, Underfitting and Model Selection


lr = LinearRegression()
lr.fit(x_train[["horsepower", "curb-weight", "engine-size", "highway-mpg"]], y_train)
yhat_train = lr.predict(x_train[["horsepower", "curb-weight", "engine-size", "highway-mpg"]])
yhat_test = lr.predict(x_test[["horsepower", "curb-weight", "engine-size", "highway-mpg"]])

import matplotlib.pyplot as plt
#%matplotlib inline
import seaborn as sns

Title = 'Distribution  Plot of  Predicted Value Using Training Data vs Training Data Distribution'
DistributionPlot(y_train, yhat_train, "Actual Values(Train)", "Predicted Values(Train)", Title)
#plt.show()

Title = 'Distribution  Plot of  Predicted Value Using Testing Data vs Testing Data Distribution'
DistributionPlot(y_test, yhat_test, "Actual Values(Test)", "Predicted Values(Test)", Title)
plt.show()

from sklearn.preprocessing import PolynomialFeatures





#Overfitting
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.45, random_state = 0)
pr = PolynomialFeatures(degree = 5)
x_train_pr = pr.fit_transform(x_train[["horsepower"]])
x_test_pr = pr.fit_transform(x_test[["horsepower"]])

poly=LinearRegression()
poly.fit(x_train_pr, y_train)

yhat = poly.predict(x_test_pr)
yhat[0:5]

print("Predicted values:", yhat[0:4])
print("True values:", y_test[0:4].values)

PollyPlot(x_train[["horsepower"]], x_test[["horsepower"]], y_train, y_test, poly, pr)

poly.score(x_train_pr, y_train)
poly.score(x_test_pr, y_test)

Rsqu_test = []

order = [1, 2, 3, 4]
for n in order:
    pr = PolynomialFeatures(degree=n)
    
    x_train_pr = pr.fit_transform(x_train[['horsepower']])
    
    x_test_pr = pr.fit_transform(x_test[['horsepower']])    
    
    lr.fit(x_train_pr, y_train)
    
    Rsqu_test.append(lr.score(x_test_pr, y_test))

plt.plot(order, Rsqu_test)
plt.xlabel('order')
plt.ylabel('R^2')
plt.title('R^2 Using Test Data')
plt.text(3, 0.75, 'Maximum R^2 ')

def f(order, test_data):
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_data, random_state=0)
    pr = PolynomialFeatures(degree=order)
    x_train_pr = pr.fit_transform(x_train[['horsepower']])
    x_test_pr = pr.fit_transform(x_test[['horsepower']])
    poly = LinearRegression()
    poly.fit(x_train_pr,y_train)
    PollyPlot(x_train[['horsepower']], x_test[['horsepower']], y_train,y_test, poly, pr)
    
#Question #4a):
#We can perform polynomial transformations with more than one feature.
#Create a "PolynomialFeatures" object "pr1" of degree two?

pr1 = PolynomialFeatures(degree=2)

#Question #4b):
#Transform the training and testing samples for the features 'horsepower',
#'curb-weight', 'engine-size' and 'highway-mpg'. Hint: use the method "fit_transform" ?

x_train_pr1=pr.fit_transform(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
x_test_pr1=pr.fit_transform(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])

#x_train_pr1 = pr1.fit_transform(x_train["horsepower", "curb-weight", "engine-size", "highway-mpg"])
#x_test_pr1 = pr1.fit_transform(x_test["horsepower", "curb-weight", "engine-size", "highway-mpg"])

#Question #4c):
#How many dimensions does the new feature have? Hint: use the attribute "shape"

print(x_train_pr1.shape)
print(x_test_pr1.shape)

#Question #4d):
#Create a linear regression model "poly1" and train the object using the method "fit" using the polynomial features?
poly1 = LinearRegression()
poly1.fit(x_train_pr1, y_train)

#Question #4e):
#Use the method "predict" to predict an output on the polynomial features, 
#then use the function "DistributionPlot" to display the distribution of the predicted output vs the test data?

yhat_test1 = poly.predict(x_test_pr1)
DistributionPlot(y_test, yhat_test1, "Actual Values (Test)", "Predicted Values (Test)", Title)

#Question #4f):
#Use the distribution plot to determine the two regions were the predicted prices are less accurate than the actual prices.

#For around 10000, predicted values > actual values
#For around 30000-40000, actual values < predicted

#Part 3: Ridge regression

pr=PolynomialFeatures(degree=2)
x_train_pr=pr.fit_transform(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg','normalized-losses','symboling']])
x_test_pr=pr.fit_transform(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg','normalized-losses','symboling']])

from sklearn.linear_model import Ridge
RigeModel=Ridge(alpha=0.1)
RigeModel.fit(x_train_pr, y_train)
yhat = RigeModel.predict(x_test_pr)
print('predicted:', yhat[0:4])
print('test set :', y_test[0:4].values)

#We select the value of Alfa that minimizes the test error, for example, we can use a for loop.
Rsqu_test = []
Rsqu_train = []
dummy1 = []
ALFA = 10 * np.array(range(0,1000))
for alfa in ALFA:
    RigeModel = Ridge(alpha=alfa) 
    RigeModel.fit(x_train_pr, y_train)
    Rsqu_test.append(RigeModel.score(x_test_pr, y_test))
    Rsqu_train.append(RigeModel.score(x_train_pr, y_train))
    
#We can plot out the value of R^2 for different Alphas
width = 12
height = 10
plt.figure(figsize=(width, height))

plt.plot(ALFA,Rsqu_test, label='validation data  ')
plt.plot(ALFA,Rsqu_train, 'r', label='training Data ')
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.legend()

Question #5):
#Perform Ridge regression and calculate the R^2 using the polynomial features, 
#use the training data to train the model and test data to test the model.#
#The parameter alpha should be set to 10.
NewRigeModel=Ridge(alpha=10)
NewRigeModel.fit(x_train_pr, y_train)
NewRigeModel.score(x_test_pr, y_test)



#Part 4: Grid Search
#The term Alfa is a hyperparameter, sklearn has the class GridSearchCV to make the process of finding the best hyperparameter simpler.

from sklearn.model_selection import GridSearchCV
NewParameters= [{'alpha': [0.001,0.1,1, 10, 100, 1000, 10000, 100000, 100000], "normalize":[True, False]}]
NewGrid = GridSearchCV(Ridge(), NewParameters, cv=4)
NewGrid.fit(x_data[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_data)
NewGrid.best_estimator_