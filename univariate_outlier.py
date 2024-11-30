import logging
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Abstract Base Class for Outlier Detection Strategy
class UnivariateOutlierDetectionStrategy(ABC):
    @abstractmethod
    def detect_outliers(self, df: pd.Series) -> pd.Series:
        pass

# Concrete Strategy for Z-Score Based Outlier Detection (Univariate)
class ZScoreOutlierDetection(UnivariateOutlierDetectionStrategy):
    def __init__(self, threshold=3):
        self.threshold = threshold

    def detect_outliers(self, df: pd.Series) -> pd.Series:
        logging.info(f"Detecting outliers using the Z-score method for {df.name}.")
        z_scores = np.abs((df - df.mean()) / df.std())
        outliers = z_scores > self.threshold
        logging.info(f"Outliers detected with Z-score threshold: {self.threshold} for {df.name}.")
        return outliers

# Concrete Strategy for IQR Based Outlier Detection (Univariate)
class IQROutlierDetection(UnivariateOutlierDetectionStrategy):
    def detect_outliers(self, df: pd.Series) -> pd.Series:
        logging.info(f"Detecting outliers using the IQR method for {df.name}.")
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        outliers = (df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))
        logging.info(f"Outliers detected using the IQR method for {df.name}.")
        return outliers

# Context Class for Univariate Outlier Detection and Handling
class UnivariateOutlierDetector:
    def __init__(self, strategy: UnivariateOutlierDetectionStrategy):
        self._strategy = strategy
        self.outlier_handling_methods = {
            "remove": self._remove_outliers,
            "cap": self._cap_outliers,
            "winsorize": self._winsorize_outliers,
            "zscore": self._zscore_outliers,
            "iqr": self._iqr_outliers
        }

    def set_strategy(self, strategy: UnivariateOutlierDetectionStrategy):
        logging.info("Switching outlier detection strategy.")
        self._strategy = strategy

    def detect_outliers(self, df: pd.Series) -> pd.Series:
        logging.info("Executing univariate outlier detection strategy.")
        return self._strategy.detect_outliers(df)

    def handle_outliers(self, df: pd.Series, method: str = "remove") -> pd.Series:
        # Detect outliers using the selected strategy
        outliers = self.detect_outliers(df)

        # Count of outliers detected
        count_outliers_before = outliers.sum()
        logging.info(f"Total outliers detected before handling: {count_outliers_before}")

        # Dynamically select the outlier handling method from the dictionary
        if method in self.outlier_handling_methods:
            return self.outlier_handling_methods[method](df, outliers)
        else:
            logging.warning(f"Unknown method '{method}'. No outlier handling performed.")
            return df

    def _remove_outliers(self, df: pd.Series, outliers: pd.Series) -> pd.Series:
        logging.info("Removing outliers from the dataset.")
        return df[~outliers]

    def _cap_outliers(self, df: pd.Series, outliers: pd.Series) -> pd.Series:
        logging.info("Capping outliers in the dataset.")
        return df.clip(lower=df.quantile(0.01), upper=df.quantile(0.99))

    def _winsorize_outliers(self, df: pd.Series, outliers: pd.Series) -> pd.Series:
        logging.info("Winsorizing outliers in the dataset.")
        lower_limit = df.quantile(0.01)
        upper_limit = df.quantile(0.99)
        return df.clip(lower=lower_limit, upper=upper_limit)

    def _zscore_outliers(self, df: pd.Series, outliers: pd.Series) -> pd.Series:
        logging.info("Handling outliers using the Z-score method.")
        z_scores = (df - df.mean()) / df.std()
        return df.clip(lower=df.mean() - 3 * df.std(), upper=df.mean() + 3 * df.std())

    def _iqr_outliers(self, df: pd.Series, outliers: pd.Series) -> pd.Series:
        logging.info("Handling outliers using the IQR method.")
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return df.clip(lower=lower_bound, upper=upper_bound)

    def visualize_outliers(self, df: pd.Series):
        logging.info(f"Visualizing outliers for feature: {df.name}")
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=df)
        plt.title(f"Boxplot of {df.name}")
        plt.show()
        logging.info("Outlier visualization completed.")

    def visualize_before_after(self, df_before: pd.Series, df_after: pd.Series):
        logging.info("Visualizing before and after outlier handling.")
        plt.figure(figsize=(12, 6))

        # Boxplot before handling
        plt.subplot(1, 2, 1)
        sns.boxplot(x=df_before)
        plt.title(f"Before Handling: {df_before.name}")

        # Boxplot after handling
        plt.subplot(1, 2, 2)
        sns.boxplot(x=df_after)
        plt.title(f"After Handling: {df_after.name}")

        plt.tight_layout()
        plt.show()
        logging.info("Before and after visualization completed.")

# Example usage
if __name__ == "__main__":
    # Load the dataset
  '''  url = 'https://raw.githubusercontent.com/analyticsindiamagazine/MocksDatasets/main/Credit_Card.csv'
    df = pd.read_csv(url)

    # Select numeric columns for univariate outlier detection
    df_numeric = df.select_dtypes(include=[np.number]).dropna()

    # Initialize the UnivariateOutlierDetector with the Z-Score based Outlier Detection Strategy
    outlier_detector = UnivariateOutlierDetector(ZScoreOutlierDetection(threshold=3))

    # Visualize outliers before handling
    for column in df_numeric.columns:
        outlier_detector.visualize_outliers(df_numeric[column])

    # Detect and handle outliers dynamically using the selected method (e.g., "iqr")
    df_cleaned = df_numeric.copy()
    for column in df_numeric.columns:
        df_cleaned[column] = outlier_detector.handle_outliers(df_numeric[column], method="iqr")

    # Visualize before and after handling outliers
    for column in df_numeric.columns:
        outlier_detector.visualize_before_after(df_numeric[column], df_cleaned[column])
'''
pass