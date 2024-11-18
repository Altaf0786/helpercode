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
from sklearn.preprocessing import LabelEncoder

# Abstract Base Class for Multivariate Analysis
class MultivariateAnalysisTemplate(ABC):
    def analyze(self, df: pd.DataFrame, target_column: str, methods=None):
        """
        Perform a comprehensive multivariate analysis on the dataset.
        Optionally, you can select specific analysis methods to run.
        """
        # Ensure that the target column is present
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in the dataset.")

        # Define a dictionary of methods
        analysis_methods = {
            'correlation_heatmap': self.generate_correlation_heatmap,
            'pairplot': self.generate_pairplot,
            'pca': self.perform_pca,
            'multivariate_regression': self.perform_multivariate_regression,
            'clustering': self.perform_clustering,
            'feature_importances': self.plot_feature_importances
        }
        
        # If no specific methods are provided, run all available methods
        if methods is None:
            methods = analysis_methods.keys()

        # Execute the specified methods
        for method in methods:
            if method in analysis_methods:
                analysis_methods[method](df, target_column)

    @abstractmethod
    def generate_correlation_heatmap(self, df: pd.DataFrame, target_column: str):
        pass

    @abstractmethod
    def generate_pairplot(self, df: pd.DataFrame, target_column: str):
        pass

    @abstractmethod
    def perform_pca(self, df: pd.DataFrame, target_column: str):
        """Perform PCA and visualize the results."""
        pass

    @abstractmethod
    def perform_multivariate_regression(self, df: pd.DataFrame, target_column: str):
        """Perform multivariate regression analysis."""
        pass

    @abstractmethod
    def perform_clustering(self, df: pd.DataFrame, target_column: str):
        """Perform clustering analysis."""
        pass

    @abstractmethod
    def plot_feature_importances(self, df: pd.DataFrame, target_column: str):
        """Plot feature importances with respect to the target variable."""
        pass


# Concrete Class for Generalized Multivariate Analysis with PCA and More
class GeneralizedMultivariateAnalysis(MultivariateAnalysisTemplate):
    def generate_correlation_heatmap(self, df: pd.DataFrame, target_column: str):
        """Generate a correlation heatmap of numerical features."""
        plt.figure(figsize=(12, 10))
        sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
        plt.title("Correlation Heatmap")
        plt.show()

    def generate_pairplot(self, df: pd.DataFrame, target_column: str):
        """Generate a pairplot of selected numerical features."""
        sns.pairplot(df)
        plt.suptitle("Pair Plot of Selected Features", y=1.02)
        plt.show()

    def perform_pca(self, df: pd.DataFrame, target_column: str):
        """Perform PCA and visualize the results."""
        features = df.drop(columns=[target_column])
        scaler = StandardScaler()
        scaled_df = scaler.fit_transform(features)
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

    def perform_multivariate_regression(self, df: pd.DataFrame, target_column: str):
        """Perform multivariate regression analysis."""
        target = df[target_column]
        features = df.drop(columns=target_column)
        model = LinearRegression()
        model.fit(features, target)
        coefficients = model.coef_

        # Plotting coefficients
        plt.figure(figsize=(10, 6))
        plt.barh(features.columns, coefficients)
        plt.xlabel('Coefficient Value')
        plt.title('Multivariate Regression Coefficients')
        plt.show()

    def perform_clustering(self, df: pd.DataFrame, target_column: str):
        """Perform clustering analysis."""
        features = df.drop(columns=[target_column])
        kmeans = KMeans(n_clusters=3)  # Specify number of clusters
        df['Cluster'] = kmeans.fit_predict(features)
        
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x=features.columns[0], y=target_column, hue='Cluster', palette='Set2', s=100)
        plt.title('K-Means Clustering Results')
        plt.xlabel(features.columns[0])
        plt.ylabel(target_column)
        plt.legend(title='Cluster')
        plt.show()

    def plot_feature_importances(self, df: pd.DataFrame, target_column: str):
        """Plot feature importances with respect to the target variable."""
        target = df[target_column]
        features = df.drop(columns=target_column)
        importances = mutual_info_regression(features, target)

        # Create a DataFrame for feature importances
        importance_df = pd.DataFrame({'Feature': features.columns, 'Importance': importances})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(data=importance_df, x='Importance', y='Feature', palette='viridis')
        plt.title(f'Feature Importances with respect to {target_column}')
        plt.show()


# Example usage
if __name__ == "__main__":
    # Load the preprocessed data
   ''' df = pd.read_csv('../extracted-data/your_data_file.csv')

    # Specify the target column
    target_column = 'SalePrice'  # Specify the target variable

    # Perform Generalized Multivariate Analysis
    multivariate_analyzer = GeneralizedMultivariateAnalysis()

    # Execute the analysis
    multivariate_analyzer.analyze(df, target_column)'''
pass    
