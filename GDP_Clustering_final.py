# -*- coding: utf-8 -*-
"""
Created on Thu May 11 11:55:13 2023

@author: yaminiperi
"""
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
import sklearn.metrics as skmet
from sklearn.preprocessing import StandardScaler

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

    plt.savefig('Clustering.png')
    
if __name__ == '__main__':
    cluster_gdp_data()

    
