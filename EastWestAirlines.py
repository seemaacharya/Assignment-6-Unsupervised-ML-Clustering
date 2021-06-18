# -*- coding: utf-8 -*-
"""
Created on Wed May 19 16:26:04 2021

@author: DELL
"""
#Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

#loading the dataset
air = pd.read_excel("EastWestAirlines.xlsx", 'data')
air.head()
air.shape

#Normalization function
def norm_func1(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)

#Normalized dataframe (considering the numerical part of the data)
df_norm1 = norm_func1(air.iloc[:,1:])
df_norm1.describe()

#Hierarchial clustering
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch # for creating the dendrogram

z1 = linkage(df_norm1, method="complete", metric="euclidean")
#plot
plt.figure(figsize=(15, 5))
plt.title("Hierarchial Clustering Dendrogram1")
plt.xlabel("Features")
plt.ylabel("Airline")
sch.dendrogram(
    z1,
    leaf_rotation=0., #rotates the x axis labels
    leaf_font_size=8., #font size for the x axis labels
)
plt.show()

air.corr()

#kmeans
#screw plot or elbow curve

k1 = list(range(2,20))
k1
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
TWSS = [] #variable for storing total within sum of squares for each kmeans
for i in k1:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm1)
    WSS = [] #variable for storing within sum of squares for each cluster
    for j in range(i):
        WSS.append(sum(cdist(df_norm1.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,df_norm1.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))
    
#scree plot
plt.figure(figsize=(20, 8))   
plt.plot(k1,TWSS, "ro-");plt.xlabel("No_of_clusters");plt.ylabel("total_within_SS");plt.xticks(k1)
air.columns

X1 = air[["Balance","Qual_miles","cc1_miles","cc2_miles","cc3_miles","Bonus_miles","Bonus_trans","Flight_miles_12mo","Flight_trans_12","Days_since_enroll","Award?"]]
clusters = KMeans(5) #5 clusters
clusters.fit(X1)
clusters.cluster_centers_
clusters.labels_
air["air_clusters"] = clusters.labels_
air.head()
air.sort_values(by=['air_clusters'],ascending = True)
X1.head()

stats1 = air.sort_values("Days_since_enroll", ascending = True)
stats1

#plot between pairs Balance~Qual_miles
sns.lmplot("Balance","Qual_miles", data = air,
           hue = "air_clusters",
           fit_reg=False, size = 6 );
#plot b/w pairs Days_since_enroll~Bonus_miles
sns.lmplot("Days_since_enroll","Bonus_miles",data=air,
           hue ="air_clusters",
           fit_reg=False, size = 6);
#Graph shows clearly (x,y) axis variables in air_clusters

#Selecting 5 clusters from the above scree plot which is the optimum numbers of clusters
model1 = KMeans(n_clusters=5)
model1.fit(df_norm1)
model1.cluster_centers_
model1.labels_

#DBSCAN
from sklearn.cluster import DBSCAN
df1=air.iloc[:,1:12]
array1 =df1.values
array1
stscaler = StandardScaler().fit(array1)
X2 = stscaler.transform(array1)
X2
dbscan = DBSCAN(eps=0.8, min_samples=6)
dbscan.fit(X2)
#Noisy samples are given the label -1.
dbscan.labels_
cl1 =pd.DataFrame(dbscan.labels_,columns=['cluster'])
cl1
pd.concat([df1,cl1],axis=1)
