import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

def load_data(filepath):
    """
    Load dataset from a CSV file.
    
    Args:
    filepath (str): Path to the CSV file.
    
    Returns:
    pd.DataFrame: Loaded dataset.
    """
    return pd.read_csv(filepath)

def impute_data(X_train, numeric_cols):
    """
    Impute missing values using mean and median.
    
    Args:
    X_train (pd.DataFrame): Training features.
    numeric_cols (list): List of numeric column names.
    
    Returns:
    pd.DataFrame: DataFrame with imputed columns.
    """
    # Calculate mean and median for each numeric column
    statistics = {}
    for col in numeric_cols:
        mean = X_train[col].mean()
        median = X_train[col].median()
        statistics[col] = {'mean': mean, 'median': median}
    
    # Impute missing values
    X_train_imputed = X_train.copy()
    for col in numeric_cols:
        X_train_imputed[f'{col}_median'] = X_train[col].fillna(statistics[col]['median'])
        X_train_imputed[f'{col}_mean'] = X_train[col].fillna(statistics[col]['mean'])
    
    return X_train_imputed, statistics

def plot_distributions_sns(X_train, X_train_imputed, numeric_cols):
    """
    Plot distributions of original and imputed columns using Seaborn.
    
    Args:
    X_train (pd.DataFrame): Original training features.
    X_train_imputed (pd.DataFrame): Training features with imputed columns.
    numeric_cols (list): List of numeric column names.
    """
    for col in numeric_cols:
        # Prepare data for plotting
        plot_df = pd.DataFrame({
            'Original': X_train[col],
            'Median Imputed': X_train_imputed[f'{col}_median'],
            'Mean Imputed': X_train_imputed[f'{col}_mean']
        })
        plot_df = plot_df.melt(var_name='Type', value_name=col)
        
        # Plot distributions using Seaborn
        plt.figure()
        sns.kdeplot(data=plot_df, x=col, hue='Type', common_norm=False, palette='tab10')
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Density')
        plt.show()

        # Plot boxplots using Seaborn
        plt.figure()
        sns.boxplot(data=plot_df, x='Type', y=col, palette='tab10')
        plt.title(f'Boxplot of {col} and its Imputed Versions')
        plt.xlabel('Type')
        plt.ylabel(col)
        plt.show()

def variance_analysis(X_train, X_train_imputed, numeric_cols):
    """
    Print variance analysis of original and imputed columns.
    
    Args:
    X_train (pd.DataFrame): Original training features.
    X_train_imputed (pd.DataFrame): Training features with imputed columns.
    numeric_cols (list): List of numeric column names.
    """
    print('Variance Analysis:')
    for col in numeric_cols:
        print(f'Original {col} variance:', X_train[col].var())
        print(f'{col} Variance after median imputation:', X_train_imputed[f'{col}_median'].var())
        print(f'{col} Variance after mean imputation:', X_train_imputed[f'{col}_mean'].var())
        print("\n")

def impute_and_analyze(df, target_col, numeric_cols, test_size=0.2, random_state=2):
    """
    Impute missing values and analyze the data.
    
    Args:
    df (pd.DataFrame): The dataset.
    target_col (str): The name of the target column.
    numeric_cols (list): List of numeric column names for imputation.
    test_size (float): Proportion of the dataset to include in the test split.
    random_state (int): Random seed for train-test split.
    
    Returns:
    pd.DataFrame, pd.DataFrame, pd.Series, pd.Series: Imputed training and test sets, target training and test sets.
    """
    # Split data into features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Identify numeric columns in X
    numeric_cols_in_X = [col for col in numeric_cols if col in X.columns]
    
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Impute data
    X_train_imputed, statistics = impute_data(X_train, numeric_cols_in_X)
    
    # Perform variance analysis
    variance_analysis(X_train, X_train_imputed, numeric_cols_in_X)
    
   
    
    # Plot distributions using Seaborn
    plot_distributions_sns(X_train, X_train_imputed, numeric_cols_in_X)
    
    # Using Sklearn for imputation
    imputer_median = SimpleImputer(strategy='median')
    imputer_mean = SimpleImputer(strategy='mean')
    
    trf = ColumnTransformer([
        ('median_imputer', imputer_median, numeric_cols_in_X),
        ('mean_imputer', imputer_mean, numeric_cols_in_X)
    ], remainder='passthrough')
    
    trf.fit(X_train)
    
    # Print statistics from the transformers
    print("Sklearn Imputer Statistics:")
    for col in numeric_cols_in_X:
        print(f'{col} Median Imputer Statistics:', trf.named_transformers_['median_imputer'].statistics_)
        print(f'{col} Mean Imputer Statistics:', trf.named_transformers_['mean_imputer'].statistics_)
        print("\n")
    
    X_train_transformed = trf.transform(X_train)
    X_test_transformed = trf.transform(X_test)
    
    return X_train_transformed, X_test_transformed, y_train, y_test

# Example usage with Titanic dataset
if __name__ == "__main__":
    """# Load Titanic dataset
    titanic_url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
    df = load_data(titanic_url)
    
    # Columns to be imputed
    numeric_cols = ['Age']
    
    # Analyze data
    X_train_transformed, X_test_transformed, y_train, y_test = impute_and_analyze(df, 'Survived', numeric_cols)
"""