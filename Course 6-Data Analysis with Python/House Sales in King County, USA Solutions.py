import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import LinearRegression
#%matplotlib inline

file_name='https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/coursera/project/kc_house_data_NaN.csv'
df=pd.read_csv(file_name)
df.head()

#Question 1
#Display the data types of each column using the attribute dtype, then take a screenshot and submit it, include your code in the image.
df.dtypes

#We use the method describe to obtain a statistical summary of the dataframe.
df.describe()

#Question 2
#Drop the columns "id" and "Unnamed: 0" from axis 1 using the method drop(),
#then use the method describe() to obtain a statistical summary of the data.
#Take a screenshot and submit it, make sure the inplace parameter is set to True
df.drop(["id", "Unnamed: 0"], axis = 1, inplace=True)
df.describe()

print("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum()) #13
print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum()) #10


#We also replace the missing values of the column 'bedrooms'/'bathrooms' with the mean of the column 'bedrooms'/'bathrooms'  
#using the method replace(). 
#Don't forget to set the  inplace  parameter top  True 
mean=df['bedrooms'].mean()
df['bedrooms'].replace(np.nan,mean, inplace=True)
mean=df['bathrooms'].mean()
df['bathrooms'].replace(np.nan,mean, inplace=True)

print("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum())
print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum())


#Question 3
#Use the method value_counts to count the number of houses with unique floor values,
#use the method .to_frame() to convert it to a dataframe.

df["floors"].value_counts().to_frame()

#Question 4
#Use the function boxplot in the seaborn library to determine whether houses with a waterfront view
#or without a waterfront view have more price outliers.

sns.boxplot(x="waterfront", y="price", data=df)

#Question 5
#Use the function regplot in the seaborn library to determine 
#if the feature sqft_above is negatively or positively correlated with price.

sns.regplot(x="sqft_above", y="price", data=df)

#We can use the Pandas method corr() to find the feature other than price that is most correlated with price.

df.corr()['price'].sort_values()





#We can Fit a linear regression model using the longitude feature 'long' and caculate the R^2.
X = df[['long']]
Y = df['price']
lm = LinearRegression()
lm.fit(X,Y)
lm.score(X, Y) #Gives R2


#Question 6
#Fit a linear regression model to predict the 'price' 
#using the feature 'sqft_living' then calculate the R^2. 
#Take a screenshot of your code and the value of the R^2.

X_2 = df[['sqft_living']]
Y_2 = df['price']
new_lm = LinearRegression()
new_lm.fit(X_2,Y_2)
new_lm.score(X_2, Y_2)

#Question 7
#Fit a linear regression model to predict the 'price' using the list of features:

features = ["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]     

#Then calculate the R^2. Take a screenshot of your code.

X_list= df[features] 
Y_3 = df['price']
new_lm_2 = LinearRegression()
new_lm_2.fit(X_list, Y_3)
new_lm_2.score(X_list, Y_3)



#Question 8
#Use the list to create a pipeline object to predict the 'price', 
#fit the object using the features in the list features, and calculate the R^2.

Input=[('scale',StandardScaler()),('polynomial', PolynomialFeatures(include_bias=False)),('model',LinearRegression())]
X_list = df[features] 
Y_3 = df['price']
pipe = Pipeline(Input)
pipe.fit(X_list , Y_3)
pipe.score(X_list, Y_3)




from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
print("done")

features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]    
X = df[features]
Y = df['price']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=1)


print("number of test samples:", x_test.shape[0])
print("number of training samples:",x_train.shape[0])

#Question 9
#Create and fit a Ridge regression object using the training data,
#set the regularization parameter to 0.1, and calculate the R^2 using the test data.
from sklearn.linear_model import Ridge

RidgeModel = Ridge(alpha = 0.1)
RidgeModel.fit(x_train, y_train)
RidgeModel.score(x_test, y_test)

#Question 10
#Perform a second order polynomial transform on both the training data and testing data. 
#Create and fit a Ridge regression object using the training data, set the regularisation parameter to 0.1, 
#and calculate the R^2 utilising the test data provided. Take a screenshot of your code and the R^2.

SecondOrderPolynomialTransform = PolynomialFeatures(degree=2)
x_train_transformed = SecondOrderPolynomialTransform.fit_transform(x_train)
x_test_transformed = SecondOrderPolynomialTransform.fit_transform(x_test)

NewRidgeModel = Ridge(alpha = 0.1)
NewRidgeModel.fit(x_train_transformed, y_train)
NewRidgeModel.score(x_test_transformed, y_test)
