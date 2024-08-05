import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from tqdm import tqdm
from datetime import datetime
from sklearn.ensemble import IsolationForest


def pipeline_standard_pca(df: pd.DataFrame, col_feature: list, k: int) -> tuple:
    """
    Creates a pipeline for applying PCA on numeric features of a DataFrame and retains original columns.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    col_feature (list): List of columns to be used as features.
    k (int): The number of principal components desired.

    Returns:
    tuple: Contains the transformed DataFrame (with original and PCA columns), 
           the PCA model, and the loadings DataFrame.
    """
    
    # Define the transformer for scaling and PCA
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[
                ('scaler', StandardScaler()),
                ('pca', PCA(n_components=k))
            ]), col_feature)
        ],
        remainder='passthrough'  # Keep other columns that are not specified in col_feature
    )

    # Fit and transform the data
    X_transformed = preprocessor.fit_transform(df)
    
    # Create DataFrame with PCA results and original features
    pca_columns = [f"PC{i+1}" for i in range(k)]
    
    # Determine which columns are kept from the original DataFrame
    original_columns = [col for col in df.columns if col not in col_feature]
    transformed_df = pd.DataFrame(X_transformed, columns=pca_columns+original_columns)
    
    # Extract the PCA model
    pca_model = preprocessor.named_transformers_['num'].named_steps['pca']
    
    # Compute loadings
    loadings = pd.DataFrame(
        pca_model.components_.T,  # Transpose to match original features
        columns=pca_columns,  # Principal components
        index=col_feature  # Original features
    )
    
    # Print explained variance
    explained_variance = pca_model.explained_variance_ratio_
    print("Explained variance by each principal component: ", explained_variance)
    print("Total variance explained: {:.2f}".format(np.sum(explained_variance)))

    return transformed_df, pca_model, loadings


def cumulative_pca_with_variance(pca_model: PCA) -> None:
    """
    Visualizes the cumulative explained variance of principal components.
    
    Parameters:
    pca_model (skPCA): The trained PCA model from scikit-learn.
    
    Returns:
    None
    """
    # Calculate individual explained variances
    individual_variances = pca_model.explained_variance_ratio_

    # Calculate cumulative explained variance
    cumulative_variances = np.cumsum(individual_variances)

    # Create a bar plot for individual explained variance
    plt.figure(figsize=(12, 7))
    bars = plt.bar(range(1, len(individual_variances) + 1), individual_variances, alpha=0.6, color='g', label='Individual Explained Variance')

    # Create a line plot for cumulative explained variance
    line = plt.plot(range(1, len(cumulative_variances) + 1), cumulative_variances, marker='o', linestyle='-', color='r', 
                    label='Cumulative Explained Variance')

    # Add percentage values above bars and points
    for i, (bar, cum_val) in enumerate(zip(bars, cumulative_variances)):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{individual_variances[i]*100:.1f}%', 
                 ha='center', va='bottom')
        plt.text(i + 1, cum_val, f'{cum_val*100:.1f}%', ha='center', va='bottom')

    # Aesthetics for the plot
    plt.xlabel('Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Explained Variance of Principal Components')
    plt.xticks(range(1, len(individual_variances) + 1))
    plt.legend(loc='upper left')
    plt.ylim(0, 1.1)  
    plt.grid(True)
    plt.show()



def defineClustering(df: pd.DataFrame, eps: float, min_samples: int, features_col: list) -> pd.DataFrame:
    """
    Defines clustering using DBSCAN on a DataFrame with specified parameters.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing the input data.
    eps (float): The maximum distance between two samples for them to be considered as in the same neighborhood.
    min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.
    features_col (list): List of column names containing the features for clustering.
    
    Returns:
    pd.DataFrame: The DataFrame with clustering predictions added.
    """
    
    # Extract features for clustering
    X = df[features_col].values
    
    # Initialize DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    
    # Fit the model and predict clusters
    df['cluster'] = dbscan.fit_predict(X)
    
    return df

def defineAnomalyDetection(df: pd.DataFrame, features_col: list) -> pd.DataFrame:
    """
    Performs anomaly detection using Isolation Forest on a DataFrame.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing the input data.
    features_col (list): List of column names containing the features for anomaly detection.
    
    Returns:
    pd.DataFrame: The DataFrame with anomaly detection results added.
    """
    
    # Extract features for anomaly detection
    X = df[features_col].values
    
    # Initialize Isolation Forest
    iso_forest = IsolationForest(contamination=0.01)  # Adjust contamination parameter as needed
    
    # Fit the model and predict anomalies
    df['anomaly'] = iso_forest.fit_predict(X)
    
    return df
