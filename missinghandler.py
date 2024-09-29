import logging
from abc import ABC, abstractmethod
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer, MissingIndicator
from sklearn.experimental import enable_iterative_imputer  # Import for IterativeImputer
from sklearn.impute import IterativeImputer


# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# Abstract Base Class for Missing Value Handling Strategy
class MissingValueHandlingStrategy(ABC):
    @abstractmethod
    def handle(self, df: pd.DataFrame, columns=None) -> pd.DataFrame:
        pass


# Concrete Strategy for KNN Imputation
class KNNImputationStrategy(MissingValueHandlingStrategy):
    def __init__(self, n_neighbors=5):
        self.imputer = KNNImputer(n_neighbors=n_neighbors)

    def handle(self, df: pd.DataFrame, columns=None) -> pd.DataFrame:
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

    def handle(self, df: pd.DataFrame, columns=None) -> pd.DataFrame:
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

    def handle(self, df: pd.DataFrame, columns=None) -> pd.DataFrame:
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
        self.indicator = MissingIndicator()

    def handle(self, df: pd.DataFrame, columns=None) -> pd.DataFrame:
        logging.info("Creating missing value indicators.")
        if columns is None:
            columns = df.columns.tolist()  # Use all columns if none are specified

        # Create a copy of the original DataFrame to avoid modifying it
        df_copy = df.copy()

        # Iterate over each specified column to create a missing indicator
        for col in columns:
            # Check if the column has any missing values
            if df_copy[col].isnull().any():
                # Fit the MissingIndicator on the specified column
                indicator_array = self.indicator.fit_transform(df_copy[[col]])
                # Create DataFrame for the indicator with the same index as original DataFrame
                indicator_df = pd.DataFrame(indicator_array, columns=[f"{col}_missing"], index=df_copy.index)
                # Concatenate the original DataFrame with the indicator DataFrame
                df_copy = pd.concat([df_copy, indicator_df], axis=1)
            else:
                logging.info(f"No missing values found in column '{col}', skipping indicator creation.")

        # Drop the original columns after creating indicators
        df_copy.drop(columns=columns, inplace=True, errors='ignore')

        logging.info("Missing value indicators created and original columns removed.")
        return df_copy


# Context Class for Handling Missing Values
class MissingValueHandler:
    def __init__(self, strategy: MissingValueHandlingStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: MissingValueHandlingStrategy):
        logging.info("Switching missing value handling strategy.")
        self._strategy = strategy

    def handle_missing_values(self, df: pd.DataFrame, columns=None) -> pd.DataFrame:
        logging.info("Executing missing value handling strategy.")
        return self._strategy.handle(df, columns)


# Example usage
if __name__ == "__main__":
   ''' # Example DataFrame
    df = pd.DataFrame({
        'A': [1, 2, None, 4],
        'B': [None, 2, 3, 4],
        'C': [1, None, None, 4],
    })

    # Initialize missing value handler with Simple Imputation for specific columns
    missing_value_handler = MissingValueHandler(SimpleImputationStrategy(strategy='mean'))
    df_imputed = missing_value_handler.handle_missing_values(df, columns=['A', 'B'])
    print("After Simple Imputation:\n", df_imputed)

    # Switch to creating Missing Indicators for specific columns
    missing_value_handler.set_strategy(MissingIndicatorStrategy())
    df_with_indicators = missing_value_handler.handle_missing_values(df, columns=['A'])
    print("After Creating Missing Indicators:\n", df_with_indicators)
'''
pass