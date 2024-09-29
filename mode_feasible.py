import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from scipy.stats import ks_2samp
from abc import ABC, abstractmethod

# Step 1: Define the Strategy Interface
# Abstract class that defines the interface for all imputation strategies
class ImputationStrategy(ABC):
    @abstractmethod
    def impute(self, df: pd.DataFrame, cat_col: str, num_col: str):
        """Impute missing values and return original and imputed data for comparison."""
        pass

# Step 2: Implement Concrete Strategies
# Concrete strategy that imputes missing values using the most frequent value (mode)
class MostFrequentImputation(ImputationStrategy):
    def impute(self, df: pd.DataFrame, cat_col: str, num_col: str):
        print(f"Imputing missing values for {cat_col} using most frequent value.")
        
        # Find the mode (most frequent value) for the categorical column
        mode_value = df[cat_col].mode()[0]
        
        # Impute missing values by filling with mode
        df[cat_col].fillna(mode_value, inplace=True)

        # Get data before and after imputation
        missing_data = df[df[cat_col].isnull()][num_col]  # Original missing data
        imputed_data = df[df[cat_col] == mode_value][num_col]  # Data for the mode after imputation
        return missing_data, imputed_data

# Concrete strategy that uses scikit-learn's SimpleImputer to fill missing values
class SklearnImputation(ImputationStrategy):
    def impute(self, df: pd.DataFrame, cat_col: str, num_col: str):
        print(f"Imputing missing values for {cat_col} using sklearn's SimpleImputer.")
        
        # Use sklearn's SimpleImputer with 'most_frequent' strategy
        imputer = SimpleImputer(strategy='most_frequent')
        df[cat_col] = imputer.fit_transform(df[[cat_col]]).ravel()

        # Get data before and after imputation
        missing_data = df[df[cat_col].isnull()][num_col]  # Original missing data
        imputed_data = df[df[cat_col] == df[cat_col].mode()[0]][num_col]  # Data for the mode after imputation
        return missing_data, imputed_data

# Step 3: Common Plotting and Statistical Testing Functions

# Function to plot value counts of a categorical column
def plot_value_counts(df, cat_col):
    """Plot value counts of a categorical column."""
    plt.figure(figsize=(10, 6))
    df[cat_col].value_counts().plot(kind='bar', title=f'{cat_col} - Value Counts')
    plt.xlabel(cat_col)
    plt.ylabel('Count')
    plt.show()

# Function to plot and compare the KDE distributions of original and imputed data
def plot_distribution_comparison(original_data, imputed_data, title):
    """Plot and compare the KDE distributions of the original and imputed data."""
    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    
    # Plot kernel density estimation (KDE) for original and imputed data
    original_data.plot(kind='kde', ax=ax, label='Original Data', color='blue')
    imputed_data.plot(kind='kde', ax=ax, label='After Imputation', color='red')
    
    # Set title and legend
    ax.set_title(title)
    ax.legend()
    plt.xlabel('SalePrice')
    plt.ylabel('Density')
    plt.show()

# Function to perform a statistical test (Kolmogorov-Smirnov test) to compare distributions
def perform_statistical_test(original_data, imputed_data):
    """Perform Kolmogorov-Smirnov test to compare distributions before and after imputation."""
    # Perform KS test to check if the original and imputed data come from the same distribution
    statistic, p_value = ks_2samp(original_data, imputed_data)
    print(f'KS Statistic: {statistic:.4f}, p-value: {p_value:.4f}')
    
    # Interpretation based on p-value
    if p_value < 0.05:
        print("The distributions are significantly different.")
    else:
        print("The distributions are not significantly different.")

# Step 4: Create a Context Class to Use Strategies

# Context class that handles the imputation process using the provided strategy
class ImputationContext:
    def __init__(self, strategy: ImputationStrategy):
        self.strategy = strategy

    # Method to change the imputation strategy dynamically
    def set_strategy(self, strategy: ImputationStrategy):
        """Change the imputation strategy."""
        self.strategy = strategy

    # Method to perform imputation and compare distributions before and after imputation
    def impute_and_compare(self, df: pd.DataFrame, cat_col: str, num_col: str):
        """Impute missing values and compare distributions."""
        
        # Plot the value counts before imputation
        print(f"Value counts before imputation for {cat_col}:")
        plot_value_counts(df, cat_col)
        
        # Perform imputation using the selected strategy
        missing_data, imputed_data = self.strategy.impute(df, cat_col, num_col)
        
        # Plot and compare the distributions before and after imputation
        print(f"Comparing distributions for {num_col} before and after imputation in {cat_col}:")
        plot_distribution_comparison(missing_data, imputed_data, f'{cat_col} - Distribution Before and After Imputation')

        # Perform a statistical test to compare the distributions
        perform_statistical_test(missing_data, imputed_data)

        # Plot the value counts after imputation
        print(f"Value counts after imputation for {cat_col}:")
        plot_value_counts(df, cat_col)

# Step 5: Implement the Main Functionality

# Main function to perform imputation and analysis using the Strategy Pattern
def main(dataset_path, use_cols, cat_cols, num_col, target_col, strategy: ImputationStrategy):
    """Main function to perform imputation and analysis using the Strategy Pattern."""
    
    # Load the dataset with specified columns
    df = pd.read_csv(dataset_path, usecols=use_cols)

    # Create an imputation context with the provided strategy
    context = ImputationContext(strategy)

    # Impute and compare distributions for each categorical column
    for feature in cat_cols:
        context.impute_and_compare(df, feature, num_col)
    
    # Return the imputed dataframe
    return df

# Example usage
if __name__ == "__main__":
   ''' # Define dataset path and columns to use
    dataset_path = 'train.csv'
    use_cols = ['GarageQual', 'FireplaceQu', 'SalePrice']
    cat_cols = ['GarageQual', 'FireplaceQu']
    num_col = 'SalePrice'
    target_col = 'SalePrice'
    
    # Use MostFrequentImputation strategy
    strategy = MostFrequentImputation()
    
    # Perform imputation using MostFrequentImputation strategy
    df_imputed = main(dataset_path, use_cols, cat_cols, num_col, target_col, strategy)
    
    # Switch to SklearnImputation strategy for comparison
    strategy = SklearnImputation()
    
    # Perform imputation using SklearnImputation strategy
    df_imputed_sklearn = main(dataset_path, use_cols, cat_cols, num_col, target_col, strategy)
'''
pass