
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 11:55:13 2023

@author: yaminiperi
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
import sklearn.metrics as skmet
from sklearn.preprocessing import StandardScaler
from scipy.optimize import curve_fit
from scipy import stats



def cluster_gdp_data():
    """
    Clusters countries based on their GDP data from 2016-2020 and saves a scatterplot of the results to a file.

    Returns
    -------
    None
    """

    # Load the data from the CSV file
    df = pd.read_csv("GDP(Current US$).csv", skiprows=4)
    df = df.drop(columns=["Unnamed: 66"])

    # drop non-numeric columns
    df = df.select_dtypes(include=['float64', 'int64'])
    
    #Tranpose the data
    df_transpose = df.T
    print(df_transpose)


    # fill missing values with the mean of the column
    df.fillna(df.mean(), inplace=True)

    # Select the columns you want to cluster on
    columns_to_cluster = ["2016", "2017", "2018", "2019", "2020"]
    df_clustering = df[columns_to_cluster]

    # Normalize the data
    scaler = StandardScaler()
    df_norm = pd.DataFrame(scaler.fit_transform(df_clustering), columns=df_clustering.columns)

    # Perform K-means clustering
    ncluster = 4
    kmeans = cluster.KMeans(n_clusters=ncluster, random_state=0)
    kmeans.fit(df_norm)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    # Calculate the silhouette score
    silhouette_avg = skmet.silhouette_score(df_norm, labels)

    # Add the cluster labels to the original dataframe
    df["Cluster"] = labels

    # Plot the clusters
    plt.figure(figsize=(12, 8))
    col = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
    for l in range(ncluster):
        plt.scatter(df["2019"][labels==l], df["2020"][labels==l], s=40, alpha=0.6, color=col[l], label=f"Cluster {l}")
    for ic in range(ncluster):
        plt.scatter(centroids[ic, 3], centroids[ic, 4], s=200, marker="*", color="black", label=f"Centroid {ic}")
    plt.xlabel("GDP 2019")
    plt.ylabel("GDP 2020")
    plt.title(f"K-Means Clustering ({ncluster} Clusters)\nSilhouette Score: {silhouette_avg:.3f}")
    plt.legend()
       
if __name__ == '__main__':
    cluster_gdp_data()
    

# Define the curve fitting model function (exponential growth)
def exp_growth(x, a, b):
    """
    Exponential growth function.

    Parameters:
    x (array): independent variable
    a (float): amplitude parameter
    b (float): growth rate parameter

    Returns:
    y (array): dependent variable
    """
    return a * np.exp(b * x)

# Define the error range function
def err_ranges(func, xdata, ydata, popt, pcov, alpha=0.05):
    """
    Calculate the error range for a fitted function.

    Parameters:
    func (function): fitted function
    xdata (array): independent variable data
    ydata (array): dependent variable data
    popt (array): optimal values for the parameters
    pcov (array): covariance matrix for the parameters
    alpha (float): significance level for the confidence interval

    Returns:
    y_err (array): upper and lower bounds of the confidence interval
    """
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

# Fit the exponential growth model to the data
popt, pcov = curve_fit(exp_growth, x_data, y_data)

# Calculate the error range for the fitted function
y_err = err_ranges(exp_growth, x_data, y_data, popt, pcov)

# Make predictions for the next 10 years
future_x = np.arange(len(gdp_data) + 10)
future_y = exp_growth(future_x, *popt)

# Calculate the confidence range for the predictions
future_y_err = err_ranges(exp_growth, future_x, future_y, popt, pcov)

# Plot the data and the fitted function with error range
plt.figure(figsize=(12, 6))
plt.plot(x_data, y_data, 'o', label='data')
plt.plot(future_x, future_y, label='fit')
plt.fill_between(future_x, future_y_err[0], future_y_err[1], alpha=0.2, label='confidence range')
plt.legend()
plt.xlabel('Year')
plt.ylabel('GDP (billions of USD)')
plt.title('Exponential growth curve fitting for United States GDP data')

# Show the plot
plt.show()
    
    

    
