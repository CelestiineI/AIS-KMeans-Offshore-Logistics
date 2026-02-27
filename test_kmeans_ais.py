import pandas as pd
from kmeans_ais import load_data, run_kmeans, visualize_clusters, elbow_method  

def test_load_data_returns_dataframe():
    df = load_data()
    #Check dataframe exists and has expected columns
    assert isinstance(df, pd.DataFrame)

    #Check not empty
    assert not df.empty

    #Check required columns
    expected_columns = ['LAT', 'LON', 'SOG', 'COG']
    for col in expected_columns:
        assert col in df.columns

def test_run_kmeans_adds_cluster_column():
    df = load_data()
    clustered_df = run_kmeans(df, k=3)  #Run K-means with 3 clusters for testing
    #Check 'Cluster' column is added
    assert 'Cluster' in clustered_df.columns

    #Check cluster labels are within expected range
    assert clustered_df['Cluster'].between(0, 2).all()  #For k=3, labels should be 0-2

    #Correct number of clusters
    assert clustered_df['Cluster'].nunique() <= 3  
