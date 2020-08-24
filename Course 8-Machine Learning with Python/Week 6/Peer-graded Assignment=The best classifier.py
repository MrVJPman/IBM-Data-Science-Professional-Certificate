from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)


from sklearn.neighbors import KNeighborsClassifier

Ks = 100
max_k = 0
max_jc_score = 0 
max_f1_score = 0

for n in range(1,Ks):
    #Train Model and Predict  
    KNearestNeighbourModel = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    y_predicted = KNearestNeighbourModel.predict(X_test)
    if max_jc_score < metrics.accuracy_score(y_test, y_predicted) and max_f1_score < metrics.f1_score(y_test, y_predicted, average="weighted"):
        max_jc_score = metrics.accuracy_score(y_test, y_predicted)
        max_f1_score = metrics.f1_score(y_test, y_predicted, average="weighted")
        max_k = n
print(max_k) #7

KNearestNeighbourModel = KNeighborsClassifier(n_neighbors = 7).fit(X_train,y_train)
y_predicted = KNearestNeighbourModel.predict(X_test)

from sklearn.metrics import jaccard_similarity_score
print("jaccard_similarity_score:", jaccard_similarity_score(y_test, y_predicted))
from sklearn.metrics import f1_score
print("f1_score: ", f1_score(y_test, y_predicted, average='weighted'))

#=================================================================

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

from sklearn.tree import DecisionTreeClassifier
DecisionTreeModel = DecisionTreeClassifier(criterion="entropy", max_depth = 1000)
DecisionTreeModel.fit(X_train,y_train)
predicted_tree = DecisionTreeModel.predict(X_test)

from sklearn.metrics import jaccard_similarity_score
print("jaccard_similarity_score:", jaccard_similarity_score(y_test, predicted_tree))
from sklearn.metrics import f1_score
print("f1_score: ", f1_score(y_test, predicted_tree, average='weighted'))






#====================Support Vector Machine====================

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

from sklearn import svm
SVMModel = svm.SVC(kernel='rbf')
SVMModel.fit(X_train, y_train) 
y_predicted = SVMModel.predict(X_test)

from sklearn.metrics import jaccard_similarity_score
print("jaccard_similarity_score:", jaccard_similarity_score(y_test, y_predicted))
from sklearn.metrics import f1_score
print("f1_score: ", f1_score(y_test, y_predicted, average='weighted'))


#====================Logistic Regression====================

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

from sklearn.linear_model import LogisticRegression
LogisticRegressionModel = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)
y_predicted = LogisticRegressionModel.predict(X_test)
y_predicted_probability = LogisticRegressionModel.predict_proba(X_test)

from sklearn.metrics import jaccard_similarity_score
print("jaccard_similarity_score:", jaccard_similarity_score(y_test, y_predicted))
from sklearn.metrics import f1_score
print("f1_score: ", f1_score(y_test, y_predicted, average='weighted'))
from sklearn.metrics import log_loss
print("log_loss: ", log_loss(y_test, y_predicted_probability))



#Model Evaluation using Test set

test_df['due_date'] = pd.to_datetime(df['due_date'])
test_df['effective_date'] = pd.to_datetime(df['effective_date'])

test_df['dayofweek'] = df['effective_date'].dt.dayofweek

test_df['weekend'] = test_df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)

test_df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)

Feature = test_df[['Principal','terms','age','Gender','weekend']]
Feature = pd.concat([Feature,pd.get_dummies(test_df['education'])], axis=1)
Feature.drop(['Master or Above'], axis = 1,inplace=True)

X = Feature
y = test_df['loan_status'].values

X = preprocessing.StandardScaler().fit(X).transform(X)




y_predicted_from_knn = KNearestNeighbourModel.predict(X)
y_predicted_tree_decision_tree = DecisionTreeModel.predict(X)
y_predicted_from_svm = SVMModel.predict(X)
y_predicted_from_LR = LogisticRegressionModel.predict(X)
y_predicted_probability_from_LR = LogisticRegressionModel.predict_proba(X)

from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score

print("============KNN============")
print("jaccard_similarity_score:", jaccard_similarity_score(y, y_predicted_from_knn))
print("f1_score: ", f1_score(y, y_predicted_from_knn, average='weighted'))

print("============Decision Tree============")
print("jaccard_similarity_score:", jaccard_similarity_score(y, y_predicted_tree_decision_tree))
print("f1_score: ", f1_score(y, y_predicted_tree_decision_tree, average='weighted'))

print("============SVM============")
print("jaccard_similarity_score:", jaccard_similarity_score(y, y_predicted_from_svm))
print("f1_score: ", f1_score(y, y_predicted_from_svm, average='weighted'))

print("============Logistic Regression============")
print("jaccard_similarity_score:", jaccard_similarity_score(y, y_predicted_from_LR))
print("f1_score: ", f1_score(y, y_predicted_from_LR, average='weighted'))
print("log_loss: ", log_loss(y, y_predicted_probability_from_LR))