import logging
from abc import ABC, abstractmethod
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Abstract Base Class for Outlier Detection Strategy
class OutlierDetectionStrategy(ABC):
    @abstractmethod
    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

# Concrete Strategy for Z-Score Based Outlier Detection
class ZScoreOutlierDetection(OutlierDetectionStrategy):
    def __init__(self, threshold=3):
        self.threshold = threshold

    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Detecting outliers using the Z-score method.")
        z_scores = np.abs((df - df.mean()) / df.std())
        outliers = z_scores > self.threshold
        logging.info(f"Outliers detected with Z-score threshold: {self.threshold}.")
        return outliers

# Concrete Strategy for IQR Based Outlier Detection
class IQROutlierDetection(OutlierDetectionStrategy):
    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Detecting outliers using the IQR method.")
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        outliers = (df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))
        logging.info("Outliers detected using the IQR method.")
        return outliers
class LOFOutlierDetection(OutlierDetectionStrategy):
    def __init__(self, n_neighbors=20):
        self.n_neighbors = n_neighbors

    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Detecting outliers using the LOF method.")
        lof = LocalOutlierFactor(n_neighbors=self.n_neighbors)
        outlier_labels = lof.fit_predict(df)
        outliers = df[outlier_labels == -1]  # Only return outliers
        logging.info(f"Detected {len(outliers)} outliers using the LOF method.")
        return outliers


# Concrete Strategy for DBSCAN Based Outlier Detection
class DBSCANOutlierDetection(OutlierDetectionStrategy):
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples

    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Detecting outliers using the DBSCAN method.")
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        cluster_labels = dbscan.fit_predict(df)
        outliers = df[cluster_labels == -1]  # Only return outliers
        logging.info(f"Detected {len(outliers)} outliers using the DBSCAN method.")
        return outliers


# Concrete Strategy for k-NN Based Outlier Detection (distance-based)
class KNNOutlierDetection(OutlierDetectionStrategy):
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Detecting outliers using the k-NN method.")
        knn = NearestNeighbors(n_neighbors=self.n_neighbors)
        knn.fit(df)
        distances, _ = knn.kneighbors(df)
        avg_distances = distances.mean(axis=1)
        threshold = np.percentile(avg_distances, 95)  # Threshold for 95th percentile
        outlier_indices = avg_distances > threshold
        outliers = df[outlier_indices]  # Only return outliers
        logging.info(f"Detected {len(outliers)} outliers using the k-NN method.")
        return outliers


# Concrete Strategy for Isolation Forest Based Outlier Detection
class IsolationForestOutlierDetection(OutlierDetectionStrategy):
    def __init__(self, contamination=0.05):
        self.contamination = contamination

    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Detecting outliers using the Isolation Forest method.")
        isolation_forest = IsolationForest(contamination=self.contamination)
        outlier_labels = isolation_forest.fit_predict(df)
        outliers = df[outlier_labels == -1]  # Only return outliers
        logging.info(f"Detected {len(outliers)} outliers using the Isolation Forest method.")
        return outliers
# Context Class for Outlier Detection and Handling
class OutlierDetector:
    def __init__(self, strategy: OutlierDetectionStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: OutlierDetectionStrategy):
        logging.info("Switching outlier detection strategy.")
        self._strategy = strategy

    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Executing outlier detection strategy.")
        return self._strategy.detect_outliers(df)

    def handle_outliers(self, df: pd.DataFrame, method="remove", **kwargs) -> pd.DataFrame:
        # Copy the original DataFrame to preserve it
        df_cleaned = df.copy()

        # Detect outliers using the selected strategy
        outliers = self.detect_outliers(df)

        # Count of outliers detected
        count_outliers_before = outliers.sum().sum()
        logging.info(f"Total outliers detected before handling: {count_outliers_before}")

        if method == "remove":
            logging.info("Removing outliers from the dataset.")
            df_cleaned = df[(~outliers).all(axis=1)]

        elif method == "cap":
            logging.info("Capping outliers in the dataset.")
            df_cleaned = df.clip(lower=df.quantile(0.01, axis=0), upper=df.quantile(0.99, axis=0), axis=1)

        elif method == "winsorize":
            logging.info("Winsorizing outliers in the dataset.")
            lower_limits = df.quantile(0.01, axis=0)
            upper_limits = df.quantile(0.99, axis=0)
            df_cleaned = df.clip(lower=lower_limits, upper=upper_limits, axis=1)
            logging.info("Winsorization completed.")
        
        elif method == "zscore":
            logging.info("Handling outliers using the Z-score method.")
            z_scores = (df - df.mean()) / df.std()
            df_cleaned = df.clip(lower=df.mean() - 3 * df.std(), upper=df.mean() + 3 * df.std(), axis=1)
            logging.info("Outliers handled using Z-score method.")

        elif method == "iqr":
            logging.info("Handling outliers using the IQR method.")
            Q1 = df.quantile(0.25)
            Q3 = df.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_cleaned = df.clip(lower=lower_bound, upper=upper_bound, axis=1)
            logging.info("Outliers handled using IQR method.")

        elif method == "impute":
            logging.info("Imputing outliers using the median.")
            # Create a copy of the original DataFrame for imputation
            median_values = df.median(axis=0)

            # Replace outliers with the median value for each column
            for column in df.columns:
                if column in df.select_dtypes(include=[np.number]).columns:  # Only apply to numeric columns
                    df_cleaned[column] = np.where(outliers[column], median_values[column], df_cleaned[column])

            logging.info("Imputation completed.")

       

        else:
            logging.warning(f"Unknown method '{method}'. No outlier handling performed.")
            return df

        return df_cleaned


    def visualize_outliers(self, df: pd.DataFrame, features: list):
        logging.info(f"Visualizing outliers for features: {features}")
        for feature in features:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=df[feature])
            plt.title(f"Boxplot of {feature}")
            plt.show()
        logging.info("Outlier visualization completed.")

    def visualize_before_after(self, df_before: pd.DataFrame, df_after: pd.DataFrame, features: list):
        logging.info("Visualizing before and after outlier handling.")
        for feature in features:
            plt.figure(figsize=(12, 6))

            # Boxplot before handling
            plt.subplot(1, 2, 1)
            sns.boxplot(x=df_before[feature])
            plt.title(f"Before Handling: {feature}")

            # Boxplot after handling
            plt.subplot(1, 2, 2)
            sns.boxplot(x=df_after[feature])
            plt.title(f"After Handling: {feature}")

            plt.tight_layout()
            plt.show()
        logging.info("Before and after visualization completed.")

# Example usage
if __name__ == "__main__":
    # Load the dataset
   ''' url = 'https://raw.githubusercontent.com/analyticsindiamagazine/MocksDatasets/main/Credit_Card.csv'
    df = pd.read_csv(url)

    # Select numeric columns for outlier detection
    df_numeric = df.select_dtypes(include=[np.number]).dropna()

    # Initialize the OutlierDetector with the Z-Score based Outlier Detection Strategy
    outlier_detector = OutlierDetector(ZScoreOutlierDetection(threshold=3))

    # Count outliers before handling
    outliers_before = outlier_detector.detect_outliers(df_numeric)
    count_outliers_before = outliers_before.sum().sum()  # Total number of outliers
    logging.info(f"Total outliers before handling: {count_outliers_before}")

    # Visualize outliers before handling
    #outlier_detector.visualize_outliers(df_numeric, features=df_numeric.columns.tolist())

    # Detect and handle outliers using Winsorization
    df_cleaned = outlier_detector.handle_outliers(df_numeric, method="iqr")

    # Count outliers after handling
    outliers_after = outlier_detector.detect_outliers(df_cleaned)
    count_outliers_after = outliers_after.sum().sum()  # Total number of outliers
    logging.info(f"Total outliers after handling: {count_outliers_after}")

    # Visualize outliers after handling
    #outlier_detector.visualize_outliers(df_cleaned, features=df_cleaned.columns.tolist())

    # Visualize before and after
    outlier_detector.visualize_before_after(df_numeric, df_cleaned, features=df_numeric.columns.tolist())'''
pass
