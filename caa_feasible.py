import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

def filter_columns_by_missing_values(df, threshold=6):
    """
    Filter columns with missing values between 0% and a given threshold.
    
    Args:
    df (pd.DataFrame): DataFrame to be analyzed.
    threshold (float): Percentage threshold for missing values.
    
    Returns:
    list: Columns with missing values between 0% and threshold.
    """
    missing_percentage = df.isnull().mean() * 100
    filtered_columns = missing_percentage[(missing_percentage > 0) & (missing_percentage < threshold)].index
    return filtered_columns

def plot_density_comparison(df, df_cca, column_name):
    """
    Plot density comparison of a column before and after CCA.
    
    Args:
    df (pd.DataFrame): Original DataFrame
    df_cca (pd.DataFrame): DataFrame after Complete Case Analysis
    column_name (str): The column name to plot
    """
    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    
    # Plot density for original data
    df[column_name].dropna().plot.density(color='red', label='Before CCA', ax=ax)
    
    # Plot density for data after CCA
    df_cca[column_name].plot.density(color='green', label='After CCA', ax=ax)
    
    plt.title(f'Density Plot Comparison for {column_name}')
    plt.xlabel(column_name)
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)  # Add grid for better readability
    plt.show()

def compare_and_decide(df, df_cca, column_name, significance_level=0.05):
    """
    Compare the distribution of a column before and after CCA using KS test.
    If the distributions are different, suggest alternative actions.
    
    Args:
    df (pd.DataFrame): Original DataFrame.
    df_cca (pd.DataFrame): DataFrame after CCA.
    column_name (str): Column to compare.
    significance_level (float): Significance level for the KS test.
    
    Returns:
    None
    """
    # Remove missing values for the original column
    original_data = df[column_name].dropna()
    cca_data = df_cca[column_name]

    # Apply Kolmogorov-Smirnov test
    statistic, p_value = ks_2samp(original_data, cca_data)

    print(f"KS Statistic for {column_name}: {statistic:.4f}, P-Value: {p_value:.4f}")
    
    # If-else logic based on the p-value
    if p_value > significance_level:
        print(f"\nThe distributions before and after CCA for '{column_name}' are similar (p > {significance_level}).")
        print("Proceed with Complete Case Analysis (CCA).\n")
        # Plot distribution comparison
        plot_density_comparison(df, df_cca, column_name)
    else:
        print(f"\nThe distributions before and after CCA for '{column_name}' are significantly different (p <= {significance_level}).")
        print("Consider alternative methods such as data imputation or other approaches instead of CCA.\n")
        # Suggest alternatives (e.g., imputation)
        print("Suggested alternatives: Mean/Median Imputation, KNN Imputation, etc.")

def compare_distributions(df, df_cca, numerical_cols, categorical_cols):
    """
    Compare distributions of numerical and categorical columns before and after CCA.
    
    Args:
    df (pd.DataFrame): Original DataFrame
    df_cca (pd.DataFrame): DataFrame after Complete Case Analysis
    numerical_cols (list): List of numerical columns
    categorical_cols (list): List of categorical columns
    """
    # Numerical Columns Comparison
    print("\n--- Numerical Columns Comparison ---")
    for col in numerical_cols:
        print(f"\nNumerical Column: {col}")

        # Mean before and after CCA
        mean_before = df[col].mean()
        mean_after = df_cca[col].mean()
        print(f"Mean Before CCA: {mean_before:.2f}, Mean After CCA: {mean_after:.2f}")

        # Apply KS Test and make decision
        compare_and_decide(df, df_cca, col)

    # Categorical Columns Comparison
    print("\n--- Categorical Columns Comparison ---")
    for col in categorical_cols:
        print(f"\nCategorical Column: {col}")

        # Percentage comparison
        temp = pd.concat([
            df[col].value_counts(normalize=True),
            df_cca[col].value_counts(normalize=True)
        ], axis=1).fillna(0)
        temp.columns = ['Original', 'CCA']
        temp = (temp * 100).round(2)  # Convert to percentage
        print(f"\nPercentage distribution for {col} before and after CCA:")
        print(temp)

def main(df):
    # Filter columns with missing values between 0% and 5%
    filtered_columns = filter_columns_by_missing_values(df)
    
    # If no columns meet the criteria, notify and exit
    if filtered_columns.empty:
        print("No columns with missing values between 0% and 5% found.")
        return

    # Drop rows with missing values in filtered columns
    df_cca = df[filtered_columns].dropna()

    # Identify numerical and categorical columns
    numerical_cols = df[filtered_columns].select_dtypes(include=['number']).columns
    categorical_cols = df[filtered_columns].select_dtypes(include=['object', 'category']).columns

    # If no columns are found, notify and exit
    if numerical_cols.empty and categorical_cols.empty:
        print("No numerical or categorical columns to compare after filtering.")
        return

    # Compare distributions column-wise
    compare_distributions(df, df_cca, numerical_cols, categorical_cols)

    # Print the shape of DataFrames
    print("Before CCA:", df.shape)
    print("After CCA:", df_cca.shape)

# Example usage with a DataFrame
if __name__ == "__main__":
    # Load your dataset
    df = pd.read_csv('/content/data_science_job.csv')
    print("Dataset Analysis:")
    main(df)import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

# Base class for strategy
class DistributionComparisonStrategy:
    """Interface for distribution comparison strategies."""
    def compare(self, df, df_cca, column_name):
        raise NotImplementedError("This method should be overridden in subclasses.")

# Concrete strategy for comparing numerical distributions using the KS test
class KSComparisonStrategy(DistributionComparisonStrategy):
    """Concrete strategy for comparing distributions using the KS test."""
    
    def compare(self, df, df_cca, column_name, significance_level=0.05):
        original_data = df[column_name].dropna()
        cca_data = df_cca[column_name]
        
        # Apply Kolmogorov-Smirnov test
        statistic, p_value = ks_2samp(original_data, cca_data)
        print(f"KS Statistic for {column_name}: {statistic:.4f}, P-Value: {p_value:.4f}")

        if p_value > significance_level:
            print(f"\nThe distributions before and after CCA for '{column_name}' are similar (p > {significance_level}).")
            print("Proceed with Complete Case Analysis (CCA).\n")
        else:
            print(f"\nThe distributions before and after CCA for '{column_name}' are significantly different (p <= {significance_level}).")
            print("Consider alternative methods such as data imputation or other approaches instead of CCA.\n")

        # Plot histogram and KDE comparison
        self.plot_histogram(df, df_cca, column_name)
        self.plot_kde(df, df_cca, column_name)

    def plot_histogram(self, df, df_cca, column_name):
        """Plot histogram comparison of a column before and after CCA."""
        plt.figure(figsize=(12, 6))
        ax = plt.gca()
        
        # Original data (before CCA)
        df[column_name].dropna().hist(bins=50, ax=ax, density=True, color='red', alpha=0.6, label='Before CCA')
        
        # Data after CCA
        df_cca[column_name].hist(bins=50, ax=ax, density=True, color='green', alpha=0.6, label='After CCA')
        
        plt.title(f'Histogram Comparison for {column_name} (Before and After CCA)')
        plt.xlabel(column_name)
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_kde(self, df, df_cca, column_name):
        """Plot KDE comparison of a column before and after CCA."""
        plt.figure(figsize=(12, 6))
        ax = plt.gca()
        
        # Original data (before CCA)
        df[column_name].dropna().plot.density(color='red', ax=ax, label='Before CCA')
        
        # Data after CCA
        df_cca[column_name].plot.density(color='green', ax=ax, label='After CCA')
        
        plt.title(f'KDE Plot Comparison for {column_name} (Before and After CCA)')
        plt.xlabel(column_name)
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)
        plt.show()

# Concrete strategy for comparing categorical distributions by percentage
class PercentageComparisonStrategy(DistributionComparisonStrategy):
    """Concrete strategy for comparing categorical distributions by percentage."""
    
    def compare(self, df, df_cca, column_name):
        temp = pd.concat([
            df[column_name].value_counts(normalize=True),
            df_cca[column_name].value_counts(normalize=True)
        ], axis=1).fillna(0)
        temp.columns = ['Original', 'CCA']
        temp = (temp * 100).round(2)  # Convert to percentage
        print(f"\nPercentage distribution for {column_name} before and after CCA:")
        print(temp)

# Function to filter columns with missing values between 0% and threshold
def filter_columns_by_missing_values(df, threshold=6):
    """Filter columns with missing values between 0% and a given threshold."""
    missing_percentage = df.isnull().mean() * 100
    filtered_columns = missing_percentage[(missing_percentage > 0) & (missing_percentage < threshold)].index
    return filtered_columns

# Compare distributions of numerical and categorical columns
def compare_distributions(df, df_cca, numerical_cols, categorical_cols):
    """Compare distributions of numerical and categorical columns before and after CCA."""
    
    # Numerical Columns Comparison
    print("\n--- Numerical Columns Comparison ---")
    ks_strategy = KSComparisonStrategy()
    for col in numerical_cols:
        print(f"\nNumerical Column: {col}")
        mean_before = df[col].mean()
        mean_after = df_cca[col].mean()
        print(f"Mean Before CCA: {mean_before:.2f}, Mean After CCA: {mean_after:.2f}")
        ks_strategy.compare(df, df_cca, col)

    # Categorical Columns Comparison
    print("\n--- Categorical Columns Comparison ---")
    percentage_strategy = PercentageComparisonStrategy()
    for col in categorical_cols:
        print(f"\nCategorical Column: {col}")
        percentage_strategy.compare(df, df_cca, col)

# Main function to run the analysis
def main(df):
    # Filter columns with missing values between 0% and 6%
    filtered_columns = filter_columns_by_missing_values(df)

    if filtered_columns.empty:
        print("No columns with missing values between 0% and 6% found.")
        return

    df_cca = df[filtered_columns].dropna()
    numerical_cols = df[filtered_columns].select_dtypes(include=['number']).columns
    categorical_cols = df[filtered_columns].select_dtypes(include=['object', 'category']).columns

    if numerical_cols.empty and categorical_cols.empty:
        print("No numerical or categorical columns to compare after filtering.")
        return

    compare_distributions(df, df_cca, numerical_cols, categorical_cols)
    print("Before CCA:", df.shape)
    print("After CCA:", df_cca.shape)

# Example to apply the strategy context on the Diabetes dataset
if __name__ == "__main__":
    # Load the diabetes datasdiabetes.csv')et (replace with your dataset path or URL)
    '''df = pd.read_csv('/content/data_science_job.csv')
    print("Dataset Analysis:")
    main(df)'''
    pass

