# Check the versions of libraries
# Python version

import sys
import scipy
import numpy as np
# import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import sklearn
from sklearn import model_selection


# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
column_headers = ['sepal-length', 'sepal-width',
                  'petal-length', 'petal-width', 'class']
df = pd.read_csv(url, names=column_headers)

data = df.values
X = data[:, 0:4]
y = data[:, 4]
# y = np.replace(y.replace({'Iris-setosa':1, 'Iris-versicolor':2, 'Iris-virginica':3})
test_size = 0.20
seed = 12345
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(
    X, y, test_size=test_size, random_state=seed)

# We will use 10-fold cross validation to estimate accuracy.
# print('X_train: {}'.format(X_train.shape))
# print('X_validation: {}'.format(X_validation.shape))
# print('Y_train: {}'.format(Y_train.shape))
# print('Y_validation: {}'.format(Y_validation.shape))

#import models

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score as accuracy

rfc = RandomForestClassifier(n_estimators=100, n_jobs=2)
rfc.fit(X_train, Y_train)
# print(rfc)

print(accuracy(Y_test, rfc.predict(X_test)))

import pickle
filename = 'finalized_model_rfc.pkl'
pickle.dump(rfc, open(filename, 'wb'))

loaded_random_forest_model = pickle.load(open('finalized_model_rfc.pkl', 'rb'))

print(accuracy(Y_test, loaded_random_forest_model.predict(X_test)))

print(X_train[0], 'first predictors')
print(Y_train[0], 'first target')

sample = [[5.1, 3.8, 1.5, 0.3], [1, 2, 3, 4], [10, 20, 30, 40]]
y_pred = rfc.predict(sample)
y_pred2 = loaded_random_forest_model.predict(sample)
print('And the value of the new flower is: ', y_pred)
print('And the value of the new flower is: ', y_pred2)

