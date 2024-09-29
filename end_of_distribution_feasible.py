import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
from sklearn.model_selection import train_test_split
from abc import ABC, abstractmethod

# Abstract Strategy for Imputation
class ImputationStrategy(ABC):
    @abstractmethod
    def calculate_imputation_values(self, column):
        """
        Calculate imputation values based on the strategy.
        
        Args:
        column (pd.Series): Column data.
        
        Returns:
        tuple: Imputation values for missing data.
        """
        pass

# Concrete Strategy 1: Normal Distribution Based Imputation
class NormalDistributionImputation(ImputationStrategy):
    def calculate_imputation_values(self, column):
        col_skew = skew(column.dropna())
        mean = column.mean()
        std = column.std()
        
        if abs(col_skew) < 0.5:  # Assuming normal distribution
            lower_impute = mean - 3 * std
            upper_impute = mean + 3 * std
        else:  # Handle as a skewed distribution
            q1 = column.quantile(0.25)
            q3 = column.quantile(0.75)
            iqr = q3 - q1
            lower_impute = q1 - 1.5 * iqr
            upper_impute = q3 + 1.5 * iqr

        return lower_impute, upper_impute

# Concrete Strategy 2: Quartile Based Imputation (For Highly Skewed Data)
class QuartileImputation(ImputationStrategy):
    def calculate_imputation_values(self, column):
        q1 = column.quantile(0.25)
        q3 = column.quantile(0.75)
        iqr = q3 - q1
        lower_impute = q1 - 1.5 * iqr
        upper_impute = q3 + 1.5 * iqr
        return lower_impute, upper_impute

# Context class that can use any imputation strategy
class DataImputationContext:
    def __init__(self, strategy: ImputationStrategy):
        self.strategy = strategy

    def calculate_imputation_values(self, column):
        return self.strategy.calculate_imputation_values(column)

# Load data function
def load_data(filepath):
    return pd.read_csv(filepath)

# Adding imputed columns
def add_imputed_columns(X_train, imputation_strategy):
    X_train_imputed = X_train.copy()
    context = DataImputationContext(imputation_strategy)

    for col in X_train.columns:
        if X_train[col].dtype in [np.float64, np.int64]:
            lower_impute, upper_impute = context.calculate_imputation_values(X_train[col])
            X_train_imputed[f'{col}_lower_imputed'] = X_train[col].fillna(lower_impute)
            X_train_imputed[f'{col}_upper_imputed'] = X_train[col].fillna(upper_impute)
    
    return X_train_imputed

# Variance analysis function
def variance_analysis(X_train, X_train_imputed, numeric_cols):
    print('Variance Analysis:')
    for col in numeric_cols:
        print(f'Original {col} variance:', X_train[col].var())
        print(f'{col} Variance after lower bound imputation:', X_train_imputed[f'{col}_lower_imputed'].var())
        print(f'{col} Variance after upper bound imputation:', X_train_imputed[f'{col}_upper_imputed'].var())
        print("\n")

# Plot distributions
def plot_distributions_sns(X_train, X_train_imputed, numeric_cols):
    for col in numeric_cols:
        plot_df = pd.DataFrame({
            'Original': X_train[col],
            'Lower Imputed': X_train_imputed[f'{col}_lower_imputed'],
            'Upper Imputed': X_train_imputed[f'{col}_upper_imputed']
        })
        plot_df = plot_df.melt(var_name='Type', value_name=col)
        
        plt.figure()
        sns.kdeplot(data=plot_df, x=col, hue='Type', common_norm=False, palette='tab10')
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Density')
        plt.show()

        plt.figure()
        sns.boxplot(data=plot_df, x='Type', y=col, palette='tab10')
        plt.title(f'Boxplot of {col} and its Imputed Versions')
        plt.xlabel('Type')
        plt.ylabel(col)
        plt.show()

# Impute and analyze function
def impute_and_analyze(df, target_col, numeric_cols, imputation_strategy, test_size=0.2, random_state=2):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    numeric_cols_in_X = [col for col in numeric_cols if col in X.columns]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    X_train_imputed = add_imputed_columns(X_train, imputation_strategy)
    
    variance_analysis(X_train, X_train_imputed, numeric_cols_in_X)
    
    plot_distributions_sns(X_train, X_train_imputed, numeric_cols_in_X)
    
    return X_train_imputed, X_test, y_train, y_test

# Example usage with Titanic dataset
if __name__ == "__main__":
    '''titanic_url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
    df = load_data(titanic_url)
    
    numeric_cols = ['Age', 'Fare']
    
    # Using NormalDistributionImputation strategy
    imputation_strategy = NormalDistributionImputation()
    
    X_train_imputed, X_test, y_train, y_test = impute_and_analyze(df, 'Survived', numeric_cols, imputation_strategy)
'''
pass