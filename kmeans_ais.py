import pandas as pd     # For loading and manipulating tabular AIS data
import numpy as np      # For numerical operations (standard in ML workflows)

from sklearn.cluster import KMeans              # K-Means clustering algorithm
from sklearn.preprocessing import StandardScaler  # Standardizes features
from sklearn.decomposition import PCA           # Principal Component Analysis

import matplotlib.pyplot as plt     # For creating plots and visualizations


# ===========================================================
# STEP 1: LOAD AND PREPARE DATA
# ===========================================================
def load_data():
    """
    Loads AIS vessel movement data from CSV file,
    selects relevant movement-related variables,
    removes incomplete records,
    and reduces dataset size for faster computation.
    """

    # Read AIS dataset
    df = pd.read_csv('AIS_2023_07_15.csv')

    # Select only key movement features:
    # LAT -> Latitude (vessel geographic position)
    # LON -> Longitude (vessel geographic position)
    # SOG -> Speed Over Ground (movement intensity)
    # COG -> Course Over Ground (movement direction)
    df = df[['LAT', 'LON', 'SOG', 'COG']].dropna()

    # Reduce dataset size for computational efficiency
    # Sampling preserves structure while improving speed
    df = df.sample(1000, random_state=42)

    return df


# ===========================================================
# STEP 2: ELBOW METHOD (CLUSTER OPTIMIZATION)
# ===========================================================
def elbow_method(data, max_k=10):
    """
    Determines the optimal number of clusters (k)
    by plotting the Within-Cluster Sum of Squares (WCSS)
    against different values of k.
    
    The 'elbow' point indicates diminishing returns.
    """

    # Standardize data so all variables contribute equally
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    wcss = []  # Stores cluster compactness values

    for i in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(scaled_data)

        # inertia_ represents WCSS (cluster compactness)
        wcss.append(kmeans.inertia_)

    # Plot WCSS curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_k + 1), wcss, marker='o')
    plt.title('Elbow Method for Optimal Clusters')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
    plt.show()


# ===========================================================
# STEP 3: PCA VARIANCE ANALYSIS (DIMENSION REDUCTION)
# ===========================================================
def pca_variance_plot(data):
    """
    Performs Principal Component Analysis (PCA)
    to determine how many dimensions are required
    to retain most of the dataset's information.
    
    The cumulative explained variance shows how much
    total variance is preserved as components increase.
    """

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    # Fit PCA without restricting components
    pca = PCA()
    pca.fit(scaled_data)

    # Compute cumulative explained variance
    cumulative_variance = pca.explained_variance_ratio_.cumsum()

    # Plot cumulative variance
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumulative_variance) + 1),
             cumulative_variance,
             marker='o')

    plt.title('PCA Cumulative Explained Variance')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.show()


# ===========================================================
# STEP 4: PCA LOADINGS (VARIABLE CONTRIBUTION ANALYSIS)
# ===========================================================
def pca_loadings(data):
    """
    Displays how strongly each original variable
    contributes to each principal component.
    
    PCA loadings help interpret what each new
    dimension represents.
    """

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    pca = PCA()
    pca.fit(scaled_data)

    # Create structured table of loadings
    loadings = pd.DataFrame(
        pca.components_,
        columns=['LAT', 'LON', 'SOG', 'COG'],
        index=[f'PC{i+1}' for i in range(len(pca.components_))]
    )

    print("\nPCA Loadings (Contribution of Each Variable):")
    print(loadings)

    return loadings


# ===========================================================
# STEP 5: RUN K-MEANS CLUSTERING
# ===========================================================
def run_kmeans(data, k=4):
    """
    Applies K-Means clustering to the standardized dataset.
    k is selected based on elbow analysis.
    """

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    kmeans = KMeans(n_clusters=k, random_state=42)

    # Assign cluster labels
    data['Cluster'] = kmeans.fit_predict(scaled_data)

    return data


# ===========================================================
# STEP 6: VISUALIZE GEOGRAPHIC CLUSTERS
# ===========================================================
def visualize_clusters(data):
    """
    Displays clustering results geographically
    using latitude and longitude.
    """

    plt.figure(figsize=(10, 6))

    plt.scatter(data['LON'],
                data['LAT'],
                c=data['Cluster'],
                cmap='viridis',
                alpha=0.6)

    plt.title('Geographic Clustering of AIS Vessel Data')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.colorbar(label='Cluster')
    plt.show()


# ===========================================================
# STEP 7: VISUALIZE CLUSTERS IN PCA REDUCED SPACE
# ===========================================================
def visualize_pca_clusters(data):
    """
    Reduces dataset to 2 principal components
    and visualizes cluster separation in reduced space.
    
    This helps evaluate cluster separability.
    """

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[['LAT', 'LON', 'SOG', 'COG']])

    pca = PCA(n_components=2)
    components = pca.fit_transform(scaled_data)

    plt.figure(figsize=(10, 6))

    plt.scatter(components[:, 0],
                components[:, 1],
                c=data['Cluster'],
                cmap='viridis',
                alpha=0.6)

    plt.title('Cluster Visualization in PCA Reduced Space')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar(label='Cluster')
    plt.show()


# ===========================================================
# MAIN EXECUTION WORKFLOW
# ===========================================================
if __name__ == "__main__":

    # 1️⃣ Load and preprocess AIS data
    df = load_data()

    # 2️⃣ Determine optimal number of clusters
    elbow_method(df)

    # 3️⃣ Analyze dimensionality reduction potential
    pca_variance_plot(df)

    # 4️⃣ Examine variable contributions
    pca_loadings(df)

    # 5️⃣ Perform clustering using selected k
    clustered_df = run_kmeans(df, k=4)

    # 6️⃣ Visualize clusters geographically
    visualize_clusters(clustered_df)

    # 7️⃣ Visualize clusters in reduced PCA space
    visualize_pca_clusters(clustered_df)

