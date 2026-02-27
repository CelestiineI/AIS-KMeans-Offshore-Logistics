import pandas as pd     #for data manipulation
import numpy as np      #for numerical operations
from sklearn.cluster import KMeans      #for K-means clustering
from sklearn.preprocessing import StandardScaler        #
import matplotlib.pyplot as plt     #for data visualization

#Step 1: Load the dataset
def load_data():
    df = pd.read_csv('AIS_2023_07_15.csv')
    #Slect important movement features
    df =df[['LAT', 'LON', 'SOG', 'COG']].dropna()  #Drop rows with missing values
    #Reduce size
    df = df.sample(1000, random_state=42)  #Use 1000 rows of the data for faster processing
    return df

#Step 2: Run K-means clustering

#Add elbow method to determine optimal number of clusters
def elbow_method(data, max_k=10):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    wcss = []
    for i in range(1, max_k+1):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(scaled_data)
        wcss.append(kmeans.inertia_)

    #Plot the elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 11), wcss, marker='o')
    plt.title('Elbow Method for Optimal Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.show()

def run_kmeans(data, k=4):
    #Preprocess the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    #Apply K-means clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
  
    #Add cluster labels to the original data
    data['Cluster'] =  kmeans.fit_predict(scaled_data)
    return data

#Step 3: Visualize the clusters
def visualize_clusters(data):
    plt.figure(figsize=(10, 6))
    plt.scatter(data['LON'], data['LAT'], c=data['Cluster'], cmap='viridis', alpha=0.5)
    plt.title('K-means Clustering of AIS Data')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.colorbar(label='Cluster')
    plt.show()

if __name__ == "__main__":
    df = load_data()   
    elbow_method(df)  #Determine optimal number of clusters using elbow method
    clustered = run_kmeans(df, k=4)  #Run K-means clustering with 4 clusters
    visualize_clusters(clustered)  #Visualize the clusters   #Load the dataset


