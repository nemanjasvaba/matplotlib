#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
=========================================================
Linear Regression Example
=========================================================
This example uses the only the first feature of the `diabetes` dataset, in
order to illustrate a two-dimensional plot of this regression technique. The
straight line can be seen in the plot, showing how linear regression attempts
to draw a straight line that will best minimize the residual sum of squares
between the observed responses in the dataset, and the responses predicted by
the linear approximation.

The coefficients, the residual sum of squares and the variance score are also
calculated.

"""
print(__doc__)


# Code source: Jaques Grobler
# License: BSD 3 clause


import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model

# Load the diabetes dataset
diabetes = datasets.load_diabetes()


# Use only one feature
diabetes_X = diabetes.data[:, np.newaxis, 2]

# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)

# The coefficients
regression_coef = regr.coef_
print('Coefficients: \n', regression_coef)
# The mean squared error
rsm_err = np.mean((regr.predict(diabetes_X_test) - diabetes_y_test) ** 2)
print("Mean squared error: %.2f" % rsm_err)
# Explained variance score: 1 is perfect prediction
var_explained = regr.score(diabetes_X_test, diabetes_y_test)
print('Variance score: %.2f' % var_explained)

# Plot outputs

plt.plot(diabetes_X_test, regr.predict(diabetes_X_test), color='blue', linewidth=3)
plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')

plt.text(0, 370, r"$\int_a^b f(x)\mathrm{d}x$",
         horizontalalignment='center', fontsize=20)


# plt.xticks(())
# plt.yticks(())

plt.show()
