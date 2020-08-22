import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
#%matplotlib inline 
import matplotlib.pyplot as plt

#Click here and press Shift+Enter
#!wget -O ChurnData.csv https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-Coursera/labs/Data_files/ChurnData.csv

churn_df = pd.read_csv("ChurnData.csv")
churn_df = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip',   'callcard', 'wireless','churn']]
churn_df['churn'] = churn_df['churn'].astype('int')

#Practice : How many rows and columns are in this dataset in total? What are the name of columns?
churn_df.shape
churn_df.columns

X = np.asarray(churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
y = np.asarray(churn_df['churn'])

#we normalize the dataset:
from sklearn import preprocessing
X = preprocessing.StandardScaler().fit(X).transform(X)

#create train/test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)

#Create Logistic Regression
#This function implements logistic regression and can use different numerical optimizers to find parameters, 
#including ‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’ solvers.
#You can find extensive information about the pros and cons of these optimizers if you search it in internet.

#The version of Logistic Regression in Scikit-learn, support regularization.
#Regularization is a technique used to solve the overfitting problem in machine learning models. 
#C parameter indicates inverse of regularization strength which must be a positive float. 
#Smaller values specify stronger regularization. Now lets fit our model with train set:
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)

yhat = LR.predict(X_test)
#predict_proba returns estimates for all classes, ordered by the label of classes. 
#So, the first column is the probability of class 1, P(Y=1|X), and second column is probability of class 0, P(Y=0|X):
yhat_prob = LR.predict_proba(X_test) 

#jaccard index : Lets try jaccard index for accuracy evaluation. 
#we can define jaccard as the size of the intersection divided
#by the size of the union of two label sets.
#If the entire set of predicted labels for a sample strictly match with the true set of labels, 
#then the subset accuracy is 1.0; otherwise it is 0.0.

from sklearn.metrics import jaccard_similarity_score
jaccard_similarity_score(y_test, yhat)

from sklearn.metrics import classification_report, confusion_matrix
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yhat, labels=[1,0])
np.set_printoptions(precision=2)


# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['churn=1','churn=0'],normalize= False,  title='Confusion matrix')

#Numerical results of confusion matrix
print (classification_report(y_test, yhat))

#log loss : Now, lets try log loss for evaluation. 
#In logistic regression, the output can be the probability of customer churn is yes (or equals to 1).
#This probability is a value between 0 and 1.
#Log loss( Logarithmic loss) measures the performance of a classifier where the predicted output is a probability value between 0 and 1.
from sklearn.metrics import log_loss
log_loss(y_test, yhat_prob)



#Practice : Try to build Logistic Regression model again for the same dataset,
#but this time, use different __solver__ and __regularization__ values? 
#What is new __logLoss__ value?
NewLogisticRegression = LogisticRegression(C=0.0001, solver='newton-cg').fit(X_train,y_train)
y_predicted_data_prob = NewLogisticRegression.predict(X_test)
log_loss(y_test, y_predicted_data_prob)

#IBM's SOLUTION
#LR2 = LogisticRegression(C=0.01, solver='sag').fit(X_train,y_train)
#yhat_prob2 = LR2.predict_proba(X_test)
#print ("LogLoss: : %.2f" % log_loss(y_test, yhat_prob2))
