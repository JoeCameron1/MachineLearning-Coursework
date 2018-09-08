#!/usr/bin/env python

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.cross_validation import KFold, cross_val_score
import copy

lr = LinearRegression(normalize=True)


# Load training and testing data
X_train = np.loadtxt('X_train.csv', delimiter=',', skiprows=1)
X_test = np.loadtxt('X_test.csv', delimiter=',', skiprows=1)
y_train = np.loadtxt('y_train.csv', delimiter=',', skiprows=1)[:,1]


poly_model = make_pipeline(PolynomialFeatures(2), Lasso(alpha=1e-10, normalize=True))


#create feature as sqrt feature1*feature2
#and remove feature 1 and 2
X_train[:,1] = np.sqrt(X_train[:,1]*X_train[:,2])
X_train = np.delete(X_train,2,1)
X_test[:,1] = np.sqrt(X_test[:,1]*X_test[:,2])
X_test = np.delete(X_test,2,1)

#remove the data that was making the model worse, ie the outliers
X_train = np.delete(X_train, [25,54,67,110,135], 0)
y_train = np.delete(y_train, [25,54,67,110,135], 0)


#leave one out cross validation
k_fold = KFold(len(y_train), n_folds=len(y_train), random_state=0)
scores = cross_val_score(poly_model, X_train, y_train, cv=k_fold, scoring='neg_mean_squared_error')
print -scores.mean()/len(y_train)


# Fit model and predict test values
y_pred = poly_model.fit(X_train, y_train).predict(X_test)

# Arrange answer in two columns. First column (with header "Id") is an
# enumeration from 0 to n-1, where n is the number of test points. Second
# column (with header "EpiOrStroma" is the predictions.
test_header = "Id,PRP"
n_points = X_test.shape[0]
y_pred_pp = np.ones((n_points, 2))
y_pred_pp[:, 0] = range(n_points)
y_pred_pp[:, 1] = y_pred
np.savetxt('my_submission.csv', y_pred_pp, fmt='%d,%f', delimiter=",",
           header=test_header, comments="")

# Note: fmt='%d' denotes that all values should be formatted as integers which
# is appropriate for classification. For regression, where the second column
# should be floating point, use fmt='%d,%f'.
