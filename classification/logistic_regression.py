from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
import sys
import collections
import copy
from sklearn.cross_validation import KFold, cross_val_score


logRegClassifier = LogisticRegression(C=0.01)
scaler = StandardScaler()

# Load training and testing data
X_train = np.loadtxt('X_train.csv', delimiter=',', skiprows=1)
X_test = np.loadtxt('X_test.csv', delimiter=',', skiprows=1)
y_train = np.loadtxt('y_train.csv', delimiter=',', skiprows=1)[:,1]

scaler.fit(X_test)
X_test = scaler.transform(X_test)
X_train = scaler.transform(X_train)

def shuffle_in_unison(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)

#shuffle the data so the order it is used will be different each time
shuffle_in_unison(y_train, X_train)


#even out data so that there is equal amounts of 2.0 and 1.0 data points
# this stops classifier predicting 2 more often
i = 0
j = 0
while i < len(y_train):
    if y_train[i]==2.0:
        y_train = np.delete(y_train, i)
        X_train = np.delete(X_train, i, 0)
        j+=1
        if j == 168:
            break
    i+=1

#ensure there is the same amount of class 1 as class 2
print collections.Counter(y_train)


# Fit model and predict test values   y = f(x)

#competition data - the values we are trying to predict
dataToPredict = X_test

#data is the input values of the fn
data = X_train

#target is the result of the fn
target = y_train


#remove feature 63 which makes the model worse
data = np.delete(data, [63], 1)
dataToPredict = np.delete(dataToPredict, [63], 1)


#cross validate the model to get accuracy
k_fold = KFold(len(target), n_folds=len(target), random_state=0)
scores = cross_val_score(logRegClassifier, data, target, cv=k_fold, n_jobs=1)
score = sum(scores)/len(scores)
print score

#new_data = scaler.fit_transform(data)
y_pred = logRegClassifier.fit(data, target).predict(dataToPredict)


# Arrange answer in two columns. First column (with header "Id") is an
# enumeration from 0 to n-1, where n is the number of test points. Second
# column (with header "EpiOrStroma" is the predictions.
test_header = "Id,EpiOrStroma"
n_points = X_test.shape[0]
y_pred_pp = np.ones((n_points, 2))
y_pred_pp[:, 0] = range(n_points)
y_pred_pp[:, 1] = y_pred
np.savetxt('my_submission.csv', y_pred_pp, fmt='%d', delimiter=",",
           header=test_header, comments="")
