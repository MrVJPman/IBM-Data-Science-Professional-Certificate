import random 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans 
from sklearn.datasets.samples_generator import make_blobs 
#%matplotlib inline

np.random.seed(0)

X, y = make_blobs(n_samples=5000, centers=[[4,4], [-2, -1], [2, -3], [1, 1]], cluster_std=0.9)

#Input
#n_samples: The total number of points equally divided among clusters. Value will be: 5000
#centers: The number of centers to generate, or the fixed center locations. Value will be: [[4, 4], [-2, -1], [2, -3],[1,1]]
#cluster_std: The standard deviation of the clusters. Value will be: 0.9

#Output
#X: Array of shape [n_samples, n_features]. (Feature Matrix) The generated samples.
#y: Array of shape [n_samples]. (Response Vector) The integer labels for cluster membership of each sample.

#Display the scatter plot of the randomly generated data.
plt.scatter(X[:, 0], X[:, 1], marker='.')

#Initialize KMeans with these parameters, where the output parameter is called k_means.
k_means = KMeans(init = "k-means++", n_clusters = 4, n_init = 12)

#init: Initialization method of the centroids. 
#k-means++: Selects initial cluster centers for k-mean clustering in a smart way to speed up convergence.
#n_clusters: The number of clusters to form as well as the number of centroids to generate.
#n_init: Number of time the k-means algorithm will be run with different centroid seeds. 
#The final results will be the best output of n_init consecutive runs in terms of inertia.

k_means.fit(X)
k_means_labels = k_means.labels_ 
#Now let's grab the labels for each point in the model using KMeans' .labels_ attribute and save it as k_means_labels
k_means_cluster_centers = k_means.cluster_centers_ #coordinates of the centers



#====================================PLOTTING==========================

## Initialize the plot with the specified dimensions.
fig = plt.figure(figsize=(6, 4))

# Colors uses a color map, which will produce an array of colors based on
# the number of labels there are. We use set(k_means_labels) to get the
# unique labels.
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_labels))))

# Create a plot
ax = fig.add_subplot(1, 1, 1)

# For loop that plots the data points and centroids.
# k will range from 0-3, which will match the possible clusters that each
# data point is in.
for k, col in zip(range(len([[4,4], [-2, -1], [2, -3], [1, 1]])), colors):

    # Create a list of all data points, where the data poitns that are 
    # in the cluster (ex. cluster 0) are labeled as true, else they are
    # labeled as false.
    my_members = (k_means_labels == k)
    
    # Define the centroid, or cluster center.
    cluster_center = k_means_cluster_centers[k]
    
    # Plots the datapoints with color col.
    ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.')
    
    # Plots the centroids with specified color, but with a darker outline
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,  markeredgecolor='k', markersize=6)

# Title of the plot
ax.set_title('KMeans')

# Remove x-axis ticks
ax.set_xticks(())

# Remove y-axis ticks
ax.set_yticks(())

# Show the plot
plt.show()

#====================================PLOTTING==========================




#Practice  : Try to cluster the above dataset into 3 clusters.

# write your code here

k_means_three_clusters = KMeans(init = "k-means++", n_clusters = 3, n_init = 12)
k_means_three_clusters.fit(X)

fig = plt.figure(figsize=(6, 4))
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_three_clusters.labels_))))
ax = fig.add_subplot(1, 1, 1)
for k, col in zip(range(len([[4,4], [-2, -1], [2, -3], [1, 1]])), colors):
    my_members = (k_means_three_clusters.labels_ == k)
    cluster_center = k_means_three_clusters.cluster_centers_[k]
    ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.')
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,  markeredgecolor='k', markersize=6)
plt.show()


#===========================Customer Segmentation with K-Means==========================

#!wget -O Cust_Segmentation.csv https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-Coursera/labs/Data_files/Cust_Segmentation.csv

import pandas as pd
cust_df = pd.read_csv("Cust_Segmentation.csv")
df = cust_df.drop('Address', axis=1) #dropped as it is categorical and provides no helpful info

#use Standard Scaler to normalize the data
from sklearn.preprocessing import StandardScaler 
X = df.values[:,1:]
X = np.nan_to_num(X)
Clus_dataSet = StandardScaler().fit_transform(X)


#modelling : 
clusterNum = 3
k_means = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 12)
k_means.fit(X)
labels = k_means.labels_


#Add the column Clus_km to the labels
df["Clus_km"] = labels

#We can easily check the centroid values by averaging the features in each cluster.
df.groupby('Clus_km').mean()
#k_means.cluster_centers_

#Now, lets look at the distribution of customers based on their age and income:
area = np.pi * ( X[:, 1])**2  
plt.scatter(X[:, 0], X[:, 3], s=area, c=labels.astype(np.float), alpha=0.5)
plt.xlabel('Age', fontsize=18)
plt.ylabel('Income', fontsize=16)

plt.show()

#3D GRAPH
from mpl_toolkits.mplot3d import Axes3D 
fig = plt.figure(1, figsize=(8, 6))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()
# plt.ylabel('Age', fontsize=18)
# plt.xlabel('Income', fontsize=16)
# plt.zlabel('Education', fontsize=16)
ax.set_xlabel('Education')
ax.set_ylabel('Age')
ax.set_zlabel('Income')

ax.scatter(X[:, 1], X[:, 0], X[:, 3], c= labels.astype(np.float))