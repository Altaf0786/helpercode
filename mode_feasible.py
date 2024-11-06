import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from scipy.stats import ks_2samp
from abc import ABC, abstractmethod

# Define the Strategy Interface
class ImputationStrategy(ABC):
    @abstractmethod
    def impute(self, df: pd.DataFrame, cat_col: str, num_col: str):
        pass

# Concrete strategy that imputes missing values using the most frequent value (mode)
class MostFrequentImputation(ImputationStrategy):
    def impute(self, df: pd.DataFrame, cat_col: str, num_col: str):
        print(f"Imputing missing values for {cat_col} using the most frequent value.")
        df = df.copy()  # Ensure we are working with a writable copy
        
        # Check for missing values in the column
        missing_data = df[df[cat_col].isnull()][num_col]  
        
        if missing_data.empty:
            print(f"No missing values found for {cat_col}.")
            return None, None  # No imputation needed if no missing values

        mode_value = df[cat_col].mode()[0]
        df[cat_col].fillna(mode_value, inplace=True)
        imputed_data = df[df[cat_col] == mode_value][num_col]
        return missing_data, imputed_data

# Concrete strategy that uses scikit-learn's SimpleImputer to fill missing values
class SklearnImputation(ImputationStrategy):
    def impute(self, df: pd.DataFrame, cat_col: str, num_col: str):
        print(f"Imputing missing values for {cat_col} using sklearn's SimpleImputer.")
        df = df.copy()  # Ensure we are working with a writable copy
        
        # Check for missing values in the column
        missing_data = df[df[cat_col].isnull()][num_col]
        
        if missing_data.empty:
            print(f"No missing values found for {cat_col}.")
            return None, None  # No imputation needed if no missing values

        imputer = SimpleImputer(strategy='most_frequent')
        df[cat_col] = imputer.fit_transform(df[[cat_col]]).ravel()
        imputed_data = df[df[cat_col] == df[cat_col].mode()[0]][num_col]
        return missing_data, imputed_data

# Plotting and Statistical Testing Functions
def plot_value_counts(df, cat_col):
    plt.figure(figsize=(10, 6))
    df[cat_col].value_counts().plot(kind='bar', title=f'{cat_col} - Value Counts')
    plt.xlabel(cat_col)
    plt.ylabel('Count')
    plt.show()

def plot_distribution_comparison(original_data, imputed_data, title):
    if original_data is None or imputed_data is None:
        print("No missing data found for imputation, skipping distribution plot.")
        return

    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    original_data.plot(kind='kde', ax=ax, label='Original Data', color='blue')
    imputed_data.plot(kind='kde', ax=ax, label='After Imputation', color='red')
    ax.set_title(title)
    ax.legend()
    plt.xlabel('Values')
    plt.ylabel('Density')
    plt.show()

def perform_statistical_test(original_data, imputed_data):
    if original_data is None or imputed_data is None:
        print("No missing data found for imputation, skipping statistical test.")
        return

    statistic, p_value = ks_2samp(original_data, imputed_data)
    print(f'KS Statistic: {statistic:.4f}, p-value: {p_value:.4f}')
    if p_value < 0.05:
        print("The distributions are significantly different.")
    else:
        print("The distributions are not significantly different.")

# Context Class
class ImputationContext:
    def __init__(self, strategy: ImputationStrategy):
        self.strategy = strategy

    def set_strategy(self, strategy: ImputationStrategy):
        self.strategy = strategy

    def impute_and_compare(self, df: pd.DataFrame, cat_col: str, num_col: str):
        print(f"Value counts before imputation for {cat_col}:")
        plot_value_counts(df, cat_col)
        missing_data, imputed_data = self.strategy.impute(df, cat_col, num_col)
        print(f"Comparing distributions for {num_col} before and after imputation in {cat_col}:")
        plot_distribution_comparison(missing_data, imputed_data, f'{cat_col} - Distribution Before and After Imputation')
        perform_statistical_test(missing_data, imputed_data)
        print(f"Value counts after imputation for {cat_col}:")
        plot_value_counts(df, cat_col)

# Main function
def main(dataset, use_cols, cat_cols, num_cols, target_col, strategy: ImputationStrategy):
    # If dataset is a file path, load the data
    if isinstance(dataset, str):
        df = pd.read_csv(dataset, usecols=use_cols)
    else:
        df = dataset
    
    context = ImputationContext(strategy)
    
    # Iterate over each feature column to apply imputation and comparison
    for feature in cat_cols:
        for num_col in num_cols:
            context.impute_and_compare(df, feature, num_col)
    
    return df

# Example usage
if __name__ == "__main__":
    # Example use
   """ use_cols = [
        'Price', 'Area', 'Location', 'No. of Bedrooms', 'Resale', 'MaintenanceStaff', 
        'Gymnasium', 'SwimmingPool', 'LandscapedGardens', 'JoggingTrack', 
        'RainWaterHarvesting', 'IndoorGames', 'ShoppingMall', 'Intercom', 
        'SportsFacility', 'ATM', 'ClubHouse', 'School', '24X7Security', 'PowerBackup', 
        'CarParking', 'StaffQuarter', 'Cafeteria', 'MultipurposeRoom', 'Hospital', 
        'WashingMachine', 'Gasconnection', 'AC', 'Wifi', "Children'splayarea", 
        'LiftAvailable', 'BED', 'VaastuCompliant', 'Microwave', 'GolfCourse', 
        'TV', 'DiningTable', 'Sofa', 'Wardrobe', 'Refrigerator', 'City'
    ]
    
    cat_cols = [
         'Resale', 'MaintenanceStaff', 'Gymnasium', 'SwimmingPool', 
        'LandscapedGardens', 'JoggingTrack', 'RainWaterHarvesting', 'IndoorGames', 
        'ShoppingMall', 'Intercom', 'SportsFacility', 'ATM', 'ClubHouse', 'School', 
        '24X7Security', 'PowerBackup', 'CarParking', 'StaffQuarter', 'Cafeteria', 
        'MultipurposeRoom', 'Hospital', 'WashingMachine', 'Gasconnection', 'AC', 
        'Wifi', "Children'splayarea", 'LiftAvailable', 'BED', 'VaastuCompliant', 
        'Microwave', 'GolfCourse', 'TV', 'DiningTable', 'Sofa', 'Wardrobe', 
        'Refrigerator', 'City'
    ]
    
    num_cols = ['Price', 'Area', 'No. of Bedrooms']
    target_col = 'Price'

    # Assuming 'data_converted' is your dataset
    # Use MostFrequentImputation strategy
    strategy = MostFrequentImputation()
    
    # Perform imputation using MostFrequentImputation strategy
    df_imputed = main(data_converted, use_cols, cat_cols, num_cols, target_col, strategy)
    
    # Switch to SklearnImputation strategy for comparison
    strategy = SklearnImputation()
    
    # Perform imputation using SklearnImputation strategy
    df_imputed_sklearn = main(data_converted, use_cols, cat_cols, num_cols, target_col, strategy)"""
pass