import pandas as pd
from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns      

def object_distinct_value(df: pd.DataFrame) -> None:
    """
    Analyzes the string (object) columns in the provided DataFrame, prints the number of distinct values
    for each column, and displays the value counts. If a column has more than 10 distinct values, only the top counts are shown.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data to analyze.

    Returns:
    - None: The function prints the number of distinct values for each column and displays value counts.
    """
    
    # Select only string (object) columns
    df = df.select_dtypes(include=['object'])
    
    # Analyze each string column
    for feature in df.columns:
        # Print the number of distinct values in the column
        print('Column ' + feature + ' has ' + str(df[feature].nunique()) + ' distinct value(s)')
        
        # Display the value counts
        if df[feature].nunique() > 10:
            display(df[feature].value_counts().head())
        else:
            display(df[feature].value_counts())
        
        # Print a newline for readability
        print('\n')

def object_distinct_value_no_value_pandas(df: pd.DataFrame) -> None:
    """
    Analyzes the string (object) columns in a Pandas DataFrame, prints the number of distinct values
    for each column, and indicates if a column is a unique key (i.e., if the number of distinct values is equal
    to the total number of rows in the DataFrame).

    Parameters:
    df (pd.DataFrame): The Pandas DataFrame to be analyzed.

    Returns:
    None
    """
    # Select only string (object) columns
    str_columns = df.select_dtypes(include=['object'])
    
    # Count the total number of rows in the DataFrame
    count_df = df.shape[0]
    print("# Dataset rows: {}".format(count_df))
    
    # For each string column, calculate and print the number of distinct values
    for feature in str_columns.columns:
        # Calculate the number of distinct values in the column
        distinct_count = str_columns[feature].nunique()
        print(f'Column {feature} has {distinct_count} distinct value(s)')
        
        # Check if the number of distinct values is equal to the total number of rows
        if count_df == distinct_count:
            print("Column is a unique key")
            
        print('\n')


def missing_data_count(df: pd.DataFrame) -> None:
    """
    Computes and prints the count and percentage of missing data for each column in the DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame for which missing data information is to be calculated.

    Returns:
    - None: The function prints a DataFrame showing the total count and percentage of missing data for each column.
    """

    # Calculate missing data
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum() / df.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    print(missing_data)

def checkMultipleValueForKey(df: pd.DataFrame, key_col: str, val_col: str) -> None:
    """
    Checks if there are multiple distinct values of a given column (`val_col`) for each unique value in another column (`key_col`).

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data to be analyzed.
    - key_col (str): The name of the column to be used as the key for grouping.
    - val_col (str): The name of the column whose distinct values are to be counted.

    Returns:
    - None: The function prints a message about the distinct values and displays the results.
    """

    # Use groupby and nunique to count distinct values of val_col for each key_col
    unique_counts = df.groupby(key_col)[val_col].nunique().reset_index()

    # Filter the results to find key_col with more than one distinct value of val_col
    multiple_values = unique_counts[unique_counts[val_col] > 1]

    # Check if there are key_col with more than one distinct value of val_col
    if multiple_values.empty:
        print("Each {} has only one value of {}.".format(key_col, val_col))
    else:
        print("There are {} with more than one value of {}:".format(key_col, val_col))
        display(multiple_values.head(5))

def visualize_correlation_matrix(df: pd.DataFrame, x_size: int = 60, y_size: int = 54) -> None:
    """
    Visualizes the correlation matrix of numeric columns in the provided DataFrame using a heatmap.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data to analyze.
    - x_size (int, optional): The width of the heatmap figure. Default is 60.
    - y_size (int, optional): The height of the heatmap figure. Default is 54.

    Returns:
    - None: The function displays a heatmap of the correlation matrix.
    """

    # Identify numeric columns
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]

    # Select only numeric columns
    df_numeric = df[numeric_cols]

    # Compute the correlation matrix
    correlation_matrix = df_numeric.corr()

    # Convert the correlation matrix into a Pandas DataFrame for visualization
    correlation_matrix_df = pd.DataFrame(correlation_matrix, index=numeric_cols, columns=numeric_cols)

    # Visualize the correlation matrix using seaborn
    plt.figure(figsize=(x_size, y_size))  # Adjust figure size as needed
    sns.heatmap(correlation_matrix_df, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Matrix')
    plt.show()


def check_drop_columns_high_correlation(df: pd.DataFrame, percent: float = 0.95) -> None:
    """
    Identifies columns in the DataFrame that have a high correlation with other columns and should be considered for removal.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data to analyze.
    - percent (float, optional): The threshold for high correlation. Columns with a correlation greater than this value will be considered for removal. Default is 0.95.

    Returns:
    - None: The function prints a list of columns that have high correlation and are candidates for removal.
    """

    # Identify numeric columns
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]

    # Select only numeric columns
    df = df[numeric_cols]
    
    # Create the absolute correlation matrix
    corr_matrix = df.corr().abs()

    # Create an upper triangular mask
    mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)

    # Apply the mask to the correlation matrix
    upper_triangle = corr_matrix.where(mask)

    # Find columns to drop based on correlation threshold
    to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > percent)]

    print("Columns to delete: ")
    print(to_drop)


def plot_violin_distribution(df: pd.DataFrame, column_name: str, target_name: str) -> None:
    """
    Plots a violin plot to compare the distribution of a specified column across different classes
    of a target variable in the DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    column_name (str): The name of the column whose distribution is to be visualized.
    target_name (str): The name of the target variable for grouping the data.

    Returns:
    None: Displays the violin plot
    """
    
    # Create a figure for the plot
    plt.figure(figsize=(10, 6))
    
    # Create the violin plot
    sns.violinplot(x=target_name, y=column_name, data=df, palette='muted')
    
    # Set the title and labels
    plt.title(f'Distribution of {column_name} by {target_name}')
    plt.xlabel(target_name)
    plt.ylabel(column_name)
    
    # Display the plot
    plt.show()


def plot_overlayed_histograms(df: pd.DataFrame, column_name: str, target_name: str) -> None:
    """
    Plots overlaid histograms to compare the distribution of a specified column across different
    classes of a target variable in the DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    column_name (str): The name of the column whose distribution is to be visualized.
    target_name (str): The name of the target variable used to group the data.

    Returns:
    None: Displays the overlaid histograms.
    """
    
    # Create a figure for the plot
    plt.figure(figsize=(10, 6))
    
    # Filter data for different classes of the target variable
    normal_cases = df[df[target_name] == 1][column_name]
    anomalies = df[df[target_name] == -1][column_name]
    
    # Plot histogram for normal cases
    sns.histplot(normal_cases, kde=True, color='blue', label='Normal Cases', alpha=0.6, bins=10)
    
    # Plot histogram for anomalies
    sns.histplot(anomalies, kde=True, color='red', label='Anomalies', alpha=0.6, bins=10)
    
    # Set the title and labels
    plt.title(f'Distribution of {column_name} by {target_name}')
    plt.xlabel(column_name)
    plt.ylabel('Frequency')
    plt.legend()
    
    # Display the plot
    plt.show()
