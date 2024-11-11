import logging
from abc import ABC, abstractmethod
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.experimental import enable_iterative_imputer  # Import for IterativeImputer
from sklearn.impute import IterativeImputer
from sklearn.impute import MissingIndicator

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# Abstract Base Class for Missing Value Handling Strategy
class MissingValueHandlingStrategy(ABC):
    @abstractmethod
    def handle(self, df: pd.DataFrame, columns=None, axis=0) -> pd.DataFrame:
        pass


# Concrete Strategy for KNN Imputation
class KNNImputationStrategy(MissingValueHandlingStrategy):
    def __init__(self, n_neighbors=5):
        self.imputer = KNNImputer(n_neighbors=n_neighbors)

    def handle(self, df: pd.DataFrame, columns=None, axis=0) -> pd.DataFrame:
        logging.info("Applying KNN imputation.")
        if columns is not None:
            imputed_array = self.imputer.fit_transform(df[columns])
            df[columns] = imputed_array
        else:
            imputed_array = self.imputer.fit_transform(df)
            df_cleaned = pd.DataFrame(imputed_array, columns=df.columns)
        logging.info("KNN imputation completed.")
        return df


# Concrete Strategy for MICE Imputation
class MICEImputationStrategy(MissingValueHandlingStrategy):
    def __init__(self):
        self.imputer = IterativeImputer()

    def handle(self, df: pd.DataFrame, columns=None, axis=0) -> pd.DataFrame:
        logging.info("Applying MICE imputation.")
        if columns is not None:
            imputed_array = self.imputer.fit_transform(df[columns])
            df[columns] = imputed_array
        else:
            imputed_array = self.imputer.fit_transform(df)
            df_cleaned = pd.DataFrame(imputed_array, columns=df.columns)
        logging.info("MICE imputation completed.")
        return df


# Concrete Strategy for Simple Imputation
class SimpleImputationStrategy(MissingValueHandlingStrategy):
    def __init__(self, strategy="mean"):
        self.imputer = SimpleImputer(strategy=strategy)

    def handle(self, df: pd.DataFrame, columns=None, axis=0) -> pd.DataFrame:
        logging.info(f"Applying Simple imputation with strategy: {self.imputer.strategy}.")
        if columns is not None:
            imputed_array = self.imputer.fit_transform(df[columns])
            df[columns] = imputed_array
        else:
            imputed_array = self.imputer.fit_transform(df)
            df_cleaned = pd.DataFrame(imputed_array, columns=df.columns)
        logging.info("Simple imputation completed.")
        return df


# Concrete Strategy for Missing Indicator from sklearn
class MissingIndicatorStrategy(MissingValueHandlingStrategy):
    def __init__(self):
        self.indicator = MissingIndicator(features="missing-only")  # Only columns with missing values

    def handle(self, df: pd.DataFrame, columns=None, axis=0) -> pd.DataFrame:
        logging.info("Creating missing value indicators for columns with missing data only.")
        
        # Fit the indicator on the DataFrame and transform it
        indicator_array = self.indicator.fit_transform(df)
        
        # Get the names of columns with missing values
        missing_columns = df.columns[self.indicator.features_]
        
        # Create a DataFrame for the indicator output with appropriate column names
        indicator_columns = [f"{col}_missing" for col in missing_columns]
        indicator_df = pd.DataFrame(indicator_array, columns=indicator_columns, index=df.index)
        
        # Convert boolean to 0 and 1 (False -> 0, True -> 1)
        indicator_df = indicator_df.astype(int)

        # Concatenate the indicators with the original DataFrame
        df_with_indicators = pd.concat([df, indicator_df], axis=1)
        
        # Drop the original columns that had missing values by name
        df_with_indicators.drop(columns=missing_columns, inplace=True)
        
        logging.info("Missing value indicators created and original columns removed.")
        return df_with_indicators
    
# Concrete Strategy for Deleting Rows/Columns Based on Missing Value Threshold
class DeleteMissingValuesStrategy(MissingValueHandlingStrategy):
    def __init__(self, axis=0, threshold=0.2):
        """
        Parameters:
        - axis: 0 for rows, 1 for columns
        - threshold: The percentage of missing values required to drop a row/column
        """
        self.axis = axis
        self.threshold = threshold

    def handle(self, df: pd.DataFrame, columns=None, axis=0, threshold=None) -> pd.DataFrame:
        if threshold is None:
            threshold = self.threshold
        
        logging.info(f"Deleting rows/columns with more than {threshold*100}% missing values.")
        
        # Calculate the threshold for missing data
        threshold_count = int(df.shape[axis] * threshold)

        # Apply deletion based on the axis (rows or columns)
        if axis == 0:  # Deleting rows
            df_cleaned = df.dropna(axis=axis, thresh=threshold_count)
        elif axis == 1:  # Deleting columns
            df_cleaned = df.dropna(axis=axis, thresh=threshold_count, how='all')

        logging.info(f"Deletion completed. Remaining rows/columns: {df_cleaned.shape[axis]}")
        return df_cleaned


# Context Class for Handling Missing Values
class MissingValueHandler:
    def __init__(self, strategy: MissingValueHandlingStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: MissingValueHandlingStrategy):
        logging.info("Switching missing value handling strategy.")
        self._strategy = strategy

    def handle_missing_values(self, df: pd.DataFrame, columns=None, axis=0) -> pd.DataFrame:
        logging.info("Executing missing value handling strategy.")
        return self._strategy.handle(df, columns, axis)

'''
# Example usage
if __name__ == "__main__":
    # Example DataFrame
    df = pd.DataFrame({
        'A': [1, 2, None, 4],
        'B': [None, 2, 3, 4],
        'C': [1, None, None, 4],
    })

    # Initialize missing value handler with Simple Imputation for specific columns
    missing_value_handler = MissingValueHandler(SimpleImputationStrategy(strategy='mean'))
    df_imputed = missing_value_handler.handle_missing_values(df, columns=['A', 'B'])
    print("After Simple Imputation:\n", df_imputed)

    # Switch to KNN Imputation
    missing_value_handler.set_strategy(KNNImputationStrategy(n_neighbors=2))
    df_imputed_knn = missing_value_handler.handle_missing_values(df, columns=['A', 'B'])
    print("\nAfter KNN Imputation:\n", df_imputed_knn)

    # Switch to MICE Imputation
    missing_value_handler.set_strategy(MICEImputationStrategy())
    df_imputed_mice = missing_value_handler.handle_missing_values(df)
    print("\nAfter MICE Imputation:\n", df_imputed_mice)

    # Switch to Missing Indicator
    missing_value_handler.set_strategy(MissingIndicatorStrategy())
    df_with_indicators = missing_value_handler.handle_missing_values(df)
    print("\nAfter Creating Missing Indicators:\n", df_with_indicators)


'''
''' The choice between axis=0 (columns) and axis=1 (rows) depends on the specific operation and context of your data, as each axis influences how missing values are handled during imputation.

1. axis=0 (Column-wise)
Imputation behavior: When you use axis=0, the operation is applied column by column.

KNN Imputation: The missing values in each column are imputed based on the nearest neighbors (other rows) in that same column. This is the most common approach for imputation, as missing values are generally assumed to be related to other rows in the same feature.
Simple Imputation: For strategies like mean or median imputation, each column is imputed with the mean or median value calculated across the non-missing values in that column.
MICE Imputation: MICE works on column-wise relationships and tries to predict the missing values for each column based on other columns. It iterates over the columns to fill in missing values using the rest of the dataset.
Why use axis=0: Column-wise imputation makes more sense when each column represents a distinct feature (e.g., different measurements for the same set of observations). Missing values in a column are likely to be better predicted using other rows (similar observations).

2. axis=1 (Row-wise)
Imputation behavior: When axis=1 is used, the operation is applied row by row.

KNN Imputation: Missing values in each row are imputed using the nearest neighbors across different columns (i.e., using other features in the same row). This method is less common and less intuitive, as it assumes missing values in the same row can be imputed based on the relationship between features (which may not always be the case).
Simple Imputation: Imputation will be done across the row, so missing values in the row will be filled with the mean, median, or most frequent value calculated across the columns for that row.
MICE Imputation: This approach is typically less effective row-wise, as MICE relies on inter-column relationships, so applying it row-wise may not yield the desired results.
Why use axis=1: This is rarely useful for imputation, as imputation typically depends on column-wise relationships (features). However, axis=1 might be used in certain situations where features (columns) are strongly dependent on each other and missingness in one row might have patterns based on other rows.

Which is better?
For Imputation:
Generally, axis=0 (column-wise) is preferred, especially when missing values are assumed to be correlated with other rows of the same column. For example, you might use this when missing data in a column (feature) is imputed based on the values of that column in other rows.
Row-wise imputation (axis=1) may not make as much sense unless there is a very specific relationship between features in a row.
Summary:
Use axis=0 for column-wise imputation, which is standard for most imputation strategies.
Use axis=1 only in cases where your rows have strong inter-feature dependencies and you believe missing values in a row can be effectively imputed based on other features within that row. This is quite rare in practice.





'''
pass