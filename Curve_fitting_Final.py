# -*- coding: utf-8 -*-
"""
Created on Wed May 10 19:13:57 2023

@author: yaminiperi
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy import stats
import matplotlib.pyplot as plt

# Define the logistic function
def logistic_function(x, a, b, c):
    return c / (1 + a * np.exp(-b*x))

# Define the error range function
def err_ranges(func, xdata, ydata, popt, pcov, alpha=0.05):
    n = len(ydata)
    p = len(popt)
    dof = max(0, n - p)
    tval = np.abs(stats.t.ppf(alpha/2, dof))
    pstd = np.sqrt(np.diag(pcov))
    y_pred = func(xdata, *popt)
    res = ydata - y_pred
    s_err = np.sqrt(np.sum(np.power(res, 2)) / dof)
    y_err = np.zeros((2, len(y_pred)))
    for i in range(len(y_pred)):
        jac = np.array([func(xdata[i], *(popt + pstd[j]*np.array([i==j for j in range(p)]))) for j in range(p)])
        y_err[0,i] = y_pred[i] - tval * s_err * np.sqrt(np.dot(jac, np.dot(pcov, jac)))
        y_err[1,i] = y_pred[i] + tval * s_err * np.sqrt(np.dot(jac, np.dot(pcov, jac)))
    return y_err

# Read the data from CSV file
data = pd.read_csv('GDP(Current US$).csv', skiprows=4)

# Extract the GDP data for a particular country
country = 'United States'
gdp_data = data.loc[data['Country Name'] == country, '1960':'2021'].values.flatten()

# Generate some sample data
x_data = np.arange(len(gdp_data))
y_data = gdp_data / 1e9  # convert to billions of dollars

# Fit the logistic function to the data
popt, pcov = curve_fit(logistic_function, x_data, y_data)

# Calculate the error range for the fitted function
y_err = err_ranges(logistic_function, x_data, y_data, popt, pcov)

# Plot the data and the fitted function with error range
plt.plot(x_data, y_data, 'o', label='data')
plt.plot(x_data, logistic_function(x_data, *popt), label='fit')
plt.fill_between(x_data, y_err[0], y_err[1], alpha=0.2, label='confidence range')
plt.legend()
plt.xlabel('Year')
plt.ylabel('GDP (billions of USD)')
plt.title('Logistic functions curve fitting for United States GDP data')


# Save the plot as a PNG file
plt.savefig('Curve_fit.png')