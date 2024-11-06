import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer

def analyze_and_impute(df: pd.DataFrame, cat_columns: list = None, num_columns: list = None):
    """
    Function to analyze and impute missing values for specified categorical and numerical columns in a dataset.
    
    Parameters:
    df (pd.DataFrame): The dataset containing the columns to analyze.
    cat_columns (list, optional): List of categorical column names to analyze. If None, all categorical columns are analyzed.
    num_columns (list, optional): List of numerical column names to analyze. If None, all numerical columns are analyzed.
    
    Returns:
    pd.DataFrame: The imputed DataFrame.
    """
    
    # If no specific columns are passed, use all categorical and numerical columns
    if cat_columns is None:
        cat_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if num_columns is None:
        num_columns = df.select_dtypes(include=['number']).columns.tolist()

    # Check if there are any categorical or numerical columns
    if len(cat_columns) == 0:
        print("No categorical columns found.")
    if len(num_columns) == 0:
        print("No numerical columns found.")
    
    # Create the SimpleImputer (to be used for imputing numerical values later)
    imputer = SimpleImputer(strategy='most_frequent')

    # For each categorical column specified
    for cat_col in cat_columns:
        # Skip columns with no missing values
        if df[cat_col].isnull().sum() == 0:
            print(f"No missing values in {cat_col}. Skipping...\n")
            continue
        
        print(f"\nAnalyzing column: {cat_col}")
        
        # 1. Percentage of missing values
        missing_percentage = df[cat_col].isnull().mean() * 100
        print(f"Percentage of Missing Values in {cat_col}: {missing_percentage:.2f}%")
        
        # 2. Value counts for the categorical column before imputation
        print(f"Value counts for {cat_col} before imputation:")
        print(df[cat_col].value_counts())
        
        # Plot the value counts before imputation
        fig, ax = plt.subplots(figsize=(12, 6))
        df[cat_col].value_counts().plot(kind='bar', ax=ax)
        ax.set_title(f"Value Counts of {cat_col} Before Imputation")
        plt.show()
        
        # 3. KDE Plots for numerical columns based on the most frequent category
        if len(num_columns) > 0:  # Check if there are any numerical columns
            for num_col in num_columns:  # Iterate through all numerical columns
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Get the most frequent category from the categorical column
                most_frequent_category = df[cat_col].mode()[0]
                
                # Plot KDE for the numerical column for the mode category
                df[df[cat_col] == most_frequent_category][num_col].plot(kind='kde', ax=ax, label=f'{num_col} with {most_frequent_category}')
                
                # Plot KDE for the numerical column for the missing values in the categorical column
                if df[cat_col].isnull().sum() > 0:
                    df[df[cat_col].isnull()][num_col].plot(kind='kde', ax=ax, color='red', label=f'{num_col} with NA')
                
                ax.legend(loc='best')
                ax.set_title(f'{cat_col} - Distribution of {num_col}')
                plt.show()
        
        # 4. Impute missing values with the most frequent category
        most_frequent_category = df[cat_col].mode()[0]
        df.loc[:, cat_col] = df[cat_col].fillna(most_frequent_category)  # Use .loc[] to ensure proper assignment
        
        # 5. Value counts for the categorical column after imputation
        print(f"Value counts for {cat_col} after imputation:")
        print(df[cat_col].value_counts())
        
        # Plot the value counts after imputation
        fig, ax = plt.subplots(figsize=(12, 6))
        df[cat_col].value_counts().plot(kind='bar', ax=ax)
        ax.set_title(f"Value Counts of {cat_col} After Imputation")
        plt.show()
        
        print(f"Completed analysis for {cat_col}\n")
    
    # Return the imputed dataframe for further analysis or usage
    return df

# Example usage:
# Replace with the actual dataset path
# df = pd.read_csv('your_data.csv')
if __name__ == "__main__":
# Call the function for specific columns in the dataset
 '''cat_columns = ['MaintenanceStaff','Gymnasium', 'SwimmingPool', 'LandscapedGardens', 'JoggingTrack',]  # Specify the categorical columns
    num_columns = ['Price']  # Specify the numerical columns
    df_imputed = analyze_and_impute(data_converted, cat_columns=cat_columns, num_columns=num_columns)'''
pass    
