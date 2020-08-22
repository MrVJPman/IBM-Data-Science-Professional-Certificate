#K Nearest Neighbour

import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
#%matplotlib inline

#!wget -O teleCust1000t.csv https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-Coursera/labs/Data_files/teleCust1000t.csv

df = pd.read_csv('teleCust1000t.csv')
df.hist(column='income', bins=50)

#To use scikit-learn library, we have to convert the Pandas data frame to a Numpy array:
X = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']] .values  #.astype(float)

#What are our labels?
y = df['custcat'].values

#Data Standardization give data zero mean and unit variance, it is good practice,
#especially for algorithms such as KNN which is based on distance of cases:

X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)

#=====================================Classification==================================
from sklearn.neighbors import KNeighborsClassifier

#Training : Lets start the algorithm with k=4 for now:
k = 4
#Train Model and Predict  
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)

#Predicting : we can use the model to predict the test set:
yhat = neigh.predict(X_test)

from sklearn import metrics
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))


#Practice : Can you build the model again, but this time with k=6?
k_is_six_neighbourhood_model = KNeighborsClassifier(n_neighbors = 6).fit(X_train,y_train)
y_predicted_data_for_k_is_six = k_is_six_neighbourhood_model.predict(X_test)

print("Train set Accuracy: ", metrics.accuracy_score(y_train, k_is_six_neighbourhood_model.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, y_predicted_data_for_k_is_six))










#======================calculating accuracy of KNN for different Ks.=========================

Ks = 10
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
ConfustionMx = [];
for n in range(1,Ks):
    
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat=neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)
    
    
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

#mean_acc-->array([0.3  , 0.29 , 0.315, 0.32 , 0.315, 0.31 , 0.335, 0.325, 0.34 ])

#Plot model accuracy for Different number of Neighbors
print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1)  
#The best accuracy was with 0.34 with k= 9