from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.feature_selection import mutual_info_regression
import numpy as np

# Abstract Base Class for Multivariate Analysis
class MultivariateAnalysisTemplate(ABC):
    def analyze(self, df: pd.DataFrame):
        """Perform a comprehensive multivariate analysis."""
        cleaned_df = self.clean_data(df)
        self.generate_correlation_heatmap(cleaned_df)
        self.generate_pairplot(cleaned_df)
        self.perform_pca(cleaned_df)
        self.perform_multivariate_regression(cleaned_df)
        self.perform_clustering(cleaned_df)
        self.plot_feature_importances(cleaned_df)

    @abstractmethod
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cleans the data before analysis."""
        pass

    @abstractmethod
    def generate_correlation_heatmap(self, df: pd.DataFrame):
        pass

    @abstractmethod
    def generate_pairplot(self, df: pd.DataFrame):
        pass

    @abstractmethod
    def perform_pca(self, df: pd.DataFrame):
        """Perform PCA and visualize the results."""
        pass

    @abstractmethod
    def perform_multivariate_regression(self, df: pd.DataFrame):
        """Perform multivariate regression analysis."""
        pass

    @abstractmethod
    def perform_clustering(self, df: pd.DataFrame):
        """Perform clustering analysis."""
        pass

    @abstractmethod
    def plot_feature_importances(self, df: pd.DataFrame):
        """Plot feature importances with respect to the target variable."""
        pass


# Concrete Class for Multivariate Analysis with PCA and More
class SimpleMultivariateAnalysis(MultivariateAnalysisTemplate):
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cleans the DataFrame by dropping non-numeric columns and handling missing values."""
        df_numeric = df.select_dtypes(include=['float64', 'int64']).dropna()
        return df_numeric

    def generate_correlation_heatmap(self, df: pd.DataFrame):
        plt.figure(figsize=(12, 10))
        sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
        plt.title("Correlation Heatmap")
        plt.show()

    def generate_pairplot(self, df: pd.DataFrame):
        sns.pairplot(df)
        plt.suptitle("Pair Plot of Selected Features", y=1.02)
        plt.show()

    def perform_pca(self, df: pd.DataFrame):
        """Perform PCA and visualize the results."""
        scaler = StandardScaler()
        scaled_df = scaler.fit_transform(df)
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled_df)
        pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
        plt.figure(figsize=(10, 7))
        plt.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.7, edgecolors='w', s=50)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('PCA of Features')
        plt.grid(True)
        plt.show()

    def perform_multivariate_regression(self, df: pd.DataFrame):
        """Perform multivariate regression analysis."""
        target = df['SalePrice']  # Change to your target variable
        features = df.drop(columns='SalePrice')  # Ensure target is excluded
        model = LinearRegression()
        model.fit(features, target)
        coefficients = model.coef_

        # Plotting coefficients
        plt.figure(figsize=(10, 6))
        plt.barh(features.columns, coefficients)
        plt.xlabel('Coefficient Value')
        plt.title('Multivariate Regression Coefficients')
        plt.show()

    def perform_clustering(self, df: pd.DataFrame):
        """Perform clustering analysis."""
        kmeans = KMeans(n_clusters=3)  # Specify number of clusters
        df['Cluster'] = kmeans.fit_predict(df)
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x='Gr Liv Area', y='SalePrice', hue='Cluster', palette='Set2', s=100)
        plt.title('K-Means Clustering Results')
        plt.xlabel('Ground Living Area')
        plt.ylabel('Sale Price')
        plt.legend(title='Cluster')
        plt.show()

    def plot_feature_importances(self, df: pd.DataFrame):
        """Plot feature importances with respect to the target variable."""
        target = df['SalePrice']  # Change to your target variable
        features = df.drop(columns='SalePrice')
        importances = mutual_info_regression(features, target)

        # Create a DataFrame for feature importances
        importance_df = pd.DataFrame({'Feature': features.columns, 'Importance': importances})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(data=importance_df, x='Importance', y='Feature', palette='viridis')
        plt.title('Feature Importances with respect to SalePrice')
        plt.show()


# Example usage
if __name__ == "__main__":
   """ # Load the data
    df = pd.read_csv('../extracted-data/your_data_file.csv')

    # Perform Multivariate Analysis
    multivariate_analyzer = SimpleMultivariateAnalysis()

    # Select important features for analysis
    selected_features = df[['SalePrice', 'Gr Liv Area', 'Overall Qual', 'Total Bsmt SF', 'Year Built']]

    # Execute the analysis
    multivariate_analyzer.analyze(selected_features)"""
pass
