from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Abstract Base Class for Multivariate Analysis
# ----------------------------------------------
class MultivariateAnalysisTemplate(ABC):
    def analyze(self, df: pd.DataFrame):
        """
        Perform a comprehensive multivariate analysis by generating a correlation heatmap,
        pair plot, and PCA plot.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data to be analyzed.

        Returns:
        None: This method orchestrates the multivariate analysis process.
        """
        self.generate_correlation_heatmap(df)
        self.generate_pairplot(df)
        self.perform_pca(df)

    @abstractmethod
    def generate_correlation_heatmap(self, df: pd.DataFrame):
        pass

    @abstractmethod
    def generate_pairplot(self, df: pd.DataFrame):
        pass

    @abstractmethod
    def perform_pca(self, df: pd.DataFrame):
        """
        Perform PCA and visualize the results.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data to be analyzed.

        Returns:
        None: Displays the PCA plot.
        """
        pass


# Concrete Class for Multivariate Analysis with PCA
# --------------------------------------------------
class SimpleMultivariateAnalysis(MultivariateAnalysisTemplate):
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
        """
        Perform PCA and visualize the results.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data to be analyzed.

        Returns:
        None: Displays the PCA plot.
        """
        # Standardize the data
        scaler = StandardScaler()
        scaled_df = scaler.fit_transform(df)

        # Apply PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled_df)

        # Create a DataFrame for PCA results
        pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])

        # Plot PCA results
        plt.figure(figsize=(10, 7))
        plt.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.7, edgecolors='w', s=50)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('PCA of Features')
        plt.grid(True)
        plt.show()


# Example usage
if __name__ == "__main__":
    # Example usage of the SimpleMultivariateAnalysis class.

    # Load the data
    # df = pd.read_csv('../extracted-data/your_data_file.csv')

    # Perform Multivariate Analysis
    # multivariate_analyzer = SimpleMultivariateAnalysis()

    # Select important features for pair plot
    # selected_features = df[['SalePrice', 'Gr Liv Area', 'Overall Qual', 'Total Bsmt SF', 'Year Built']]

    # Execute the analysis
    # multivariate_analyzer.analyze(selected_features)
    pass
