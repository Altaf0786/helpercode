import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
# Outlier Detection Strategy Interface
class OutlierDetectionStrategy:
    def detect(self, df, numerical_columns, **kwargs):
        raise NotImplementedError("Subclasses should implement this method!")

# One-Class SVM Strategy
class OneClassSVMStrategy(OutlierDetectionStrategy):
    def detect(self, df, numerical_columns, **kwargs):
        svm = OneClassSVM(**kwargs)
        df['Outlier_SVM'] = svm.fit_predict(df[numerical_columns])
        df['Outlier_SVM'] = df['Outlier_SVM'].apply(lambda x: 1 if x == -1 else 0)
        return df

# KNN Strategy
class KNNStrategy(OutlierDetectionStrategy):
    def detect(self, df, numerical_columns, **kwargs):
        knn = NearestNeighbors(**kwargs)
        knn.fit(df[numerical_columns])
        distances, indices = knn.kneighbors(df[numerical_columns])
        avg_distances = distances.mean(axis=1)
        threshold = np.percentile(avg_distances, 95)
        df['Outlier_KNN'] = (avg_distances > threshold).astype(int)
        return df

# Isolation Forest Strategy
class IsolationForestStrategy(OutlierDetectionStrategy):
    def detect(self, df, numerical_columns, **kwargs):
        iso_forest = IsolationForest(**kwargs)
        df['Outlier_IF'] = iso_forest.fit_predict(df[numerical_columns])
        df['Outlier_IF'] = df['Outlier_IF'].apply(lambda x: 1 if x == -1 else 0)
        return df

# LOF Strategy
class LOFStrategy(OutlierDetectionStrategy):
    def detect(self, df, numerical_columns, **kwargs):
        lof = LocalOutlierFactor(**kwargs)
        df['Outlier_LOF'] = lof.fit_predict(df[numerical_columns])
        df['Outlier_LOF'] = df['Outlier_LOF'].apply(lambda x: 1 if x == -1 else 0)
        return df

# DBSCAN Strategy
class DBSCANStrategy(OutlierDetectionStrategy):
    def detect(self, df, numerical_columns, **kwargs):
        dbscan = DBSCAN(**kwargs)
        df['Outlier_DBSCAN'] = dbscan.fit_predict(df[numerical_columns])
        df['Outlier_DBSCAN'] = df['Outlier_DBSCAN'].apply(lambda x: 1 if x == -1 else 0)
        return df

# Outlier Detection Class with Method-wise Handling
class OutlierDetection:
    def __init__(self, strategy: OutlierDetectionStrategy, df, numerical_columns):
        self.strategy = strategy
        self.df = df
        self.numerical_columns = numerical_columns

    def detect_outliers(self, **kwargs):
        return self.strategy.detect(self.df, self.numerical_columns, **kwargs)

    def handle_outliers(self, method='remove', outlier_column=None):
        if method == 'remove':
            # Remove outliers method-wise
            df_no_outliers = self.df[self.df[outlier_column] == 0]
            return df_no_outliers
        elif method == 'impute':
            # Impute outliers with median method-wise
            df_imputed = self.df.copy()
            median = df_imputed[self.numerical_columns].median()
            df_imputed.loc[df_imputed[outlier_column] == 1, self.numerical_columns] = median
            return df_imputed
        elif method == 'flag':
            # Flagging outliers
            return self.df.copy()

    def visualize_outlier_distributions(self):
        plt.figure(figsize=(15, 12))

        # Define outlier columns based on the strategy used
        if isinstance(self.strategy, KNNStrategy):
            outlier_columns = ['Outlier_KNN']
        elif isinstance(self.strategy, OneClassSVMStrategy):
            outlier_columns = ['Outlier_SVM']
        elif isinstance(self.strategy, DBSCANStrategy):
            outlier_columns = ['Outlier_DBSCAN']
        elif isinstance(self.strategy, LOFStrategy):
            outlier_columns = ['Outlier_LOF']
        elif isinstance(self.strategy, IsolationForestStrategy):
            outlier_columns = ['Outlier_IF']
        else:
            # Default case if needed, plotting all methods
            outlier_columns = ['Outlier_SVM', 'Outlier_KNN', 'Outlier_DBSCAN', 'Outlier_LOF', 'Outlier_IF']

        # Plot the outlier distributions
        for i, col in enumerate(outlier_columns, 1):
            plt.subplot(1, len(outlier_columns), i)
            self.df[col].value_counts().plot(kind='bar', color=['lightblue', 'salmon'])
            plt.xlabel('Outlier Class')
            plt.ylabel('Count')
            plt.title(f'{col} Outlier Detection')
            plt.xticks([0, 1], ['Normal', 'Outlier'], rotation=0)

        plt.tight_layout()
        plt.show()

        # Display count and percentage of outliers
        for col in outlier_columns:
            outlier_count = self.df[col].sum()
            normal_count = len(self.df) - outlier_count
            outlier_percentage = (outlier_count / len(self.df)) * 100
            print(f"\n{col}:")
            print(f"  Outliers Count: {outlier_count} ({outlier_percentage:.2f}%)")
            print(f"  Normal Count: {normal_count} ({100 - outlier_percentage:.2f}%)")

    def visualize_pca_with_colors(self, df, outlier_column, title):
        # PCA for 2D visualization
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(df[self.numerical_columns])

        # Assign colors for outliers and non-outliers
        colors = df[outlier_column].apply(lambda x: 'red' if x == 1 else 'blue')

        # Scatter plot
        plt.figure(figsize=(7, 7))
        plt.scatter(pca_result[:, 0], pca_result[:, 1], c=colors, s=10, alpha=0.7)
        plt.title(title)
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')
        plt.legend(handles=[
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Non-Outlier'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Outlier')
        ])


    ''' def visualize_tsne_with_colors(self, df, outlier_column, title):
        """
        Visualize t-SNE results with specified colors for different labels and additional error counts in the title.
        
        Parameters:
            df (pd.DataFrame): The dataframe containing numerical columns for t-SNE and outlier information.
            outlier_column (str): The column indicating outliers (e.g., 0 for normal, 1 for outlier).
            title (str): Title for the plot.
        """
        # Perform t-SNE for 2D visualization
        tsne = TSNE(n_components=2, random_state=42)
        tsne_result = tsne.fit_transform(df[self.numerical_columns])

        # Define colors and labels
        colors = ["blue", "red"]  # Blue for non-outliers, red for outliers
        labels = ["Non-Outlier", "Outlier"]

        # Scatter plot
        plt.figure(figsize=(10, 7))
        for i, color in enumerate(colors):
            mask = df[outlier_column] == i
            plt.scatter(
                tsne_result[mask, 0],
                tsne_result[mask, 1],
                c=color,
                edgecolors="k",
                label=labels[i],
                s=50,  # Adjust size as needed
                alpha=0.7
            )

        # Plot title and labels
        plt.title(title)
        plt.legend(loc="upper right")
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")
        plt.grid(True)
        plt.tight_layout()
        plt.show()'''


if __name__ == "__main__":
        '''   # Load Dataset
            url = 'https://raw.githubusercontent.com/analyticsindiamagazine/MocksDatasets/main/Credit_Card.csv'
            df = pd.read_csv(url)

            # Select numeric columns for outlier detection
            df_numeric = df.select_dtypes(include=[np.number]).dropna()
            numerical_columns = df_numeric.columns.tolist()

            # Example: Apply KNN strategy to detect outliers
            knn_strategy = KNNStrategy()
            outlier_detector = OutlierDetection(strategy=knn_strategy, df=df, numerical_columns=numerical_columns)

            # Detect outliers with KNN
            df_with_outliers = outlier_detector.detect_outliers(n_neighbors=5)

            # Visualize the outlier distributions for KNN
            outlier_detector.visualize_outlier_distributions()

            # Handle and visualize KNN method-wise:
            df_no_outliers_knn = outlier_detector.handle_outliers(method='remove', outlier_column='Outlier_KNN')
            outlier_detector.visualize_pca_with_colors(df=df, outlier_column='Outlier_KNN', title="PCA - KNN Outlier Detection")

            # Example: Apply One-Class SVM strategy to detect outliers
            svm_strategy = OneClassSVMStrategy()
            outlier_detector_svm = OutlierDetection(strategy=svm_strategy, df=df, numerical_columns=numerical_columns)

            # Detect outliers with One-Class SVM
            df_with_outliers_svm = outlier_detector_svm.detect_outliers(nu=0.1)

            # Visualize the outlier distributions for SVM
            outlier_detector_svm.visualize_outlier_distributions()

            # Handle and visualize One-Class SVM method-wise:
            df_no_outliers_svm = outlier_detector_svm.handle_outliers(method='remove', outlier_column='Outlier_SVM')
            outlier_detector_svm.visualize_pca_with_colors(df=df, outlier_column='Outlier_SVM', title="PCA - One-Class SVM Outlier Detection")
            


            # Example: Apply Isolation Forest strategy to detect outliers
            iso_forest_strategy = IsolationForestStrategy()
            outlier_detector_if = OutlierDetection(strategy=iso_forest_strategy, df=df, numerical_columns=numerical_columns)

            # Detect outliers with Isolation Forest
            df_with_outliers_if = outlier_detector_if.detect_outliers(contamination=0.05)

            # Visualize the outlier distributions for Isolation Forest
            outlier_detector_if.visualize_outlier_distributions()

            # Handle and visualize Isolation Forest method-wise:
            df_no_outliers_if = outlier_detector_if.handle_outliers(method='remove', outlier_column='Outlier_IF')
            outlier_detector_if.visualize_pca_with_colors(df=df, outlier_column='Outlier_IF', title="PCA - Isolation Forest Outlier Detection")

            # Example: Apply LOF strategy to detect outliers
            lof_strategy = LOFStrategy()
            outlier_detector_lof = OutlierDetection(strategy=lof_strategy, df=df, numerical_columns=numerical_columns)

            # Detect outliers with LOF
            df_with_outliers_lof = outlier_detector_lof.detect_outliers(n_neighbors=20)

            # Visualize the outlier distributions for LOF
            outlier_detector_lof.visualize_outlier_distributions()

            # Handle and visualize LOF method-wise:
            df_no_outliers_lof = outlier_detector_lof.handle_outliers(method='remove', outlier_column='Outlier_LOF')
            outlier_detector_lof.visualize_pca_with_colors(df=df, outlier_column='Outlier_LOF', title="PCA - LOF Outlier Detection")

            # Example: Apply DBSCAN strategy to detect outliers
            dbscan_strategy = DBSCANStrategy()
            outlier_detector_dbscan = OutlierDetection(strategy=dbscan_strategy, df=df, numerical_columns=numerical_columns)

            # Detect outliers with DBSCAN
            df_with_outliers_dbscan = outlier_detector_dbscan.detect_outliers(eps=0.5, min_samples=5)

            # Visualize the outlier distributions for DBSCAN
            outlier_detector_dbscan.visualize_outlier_distributions()

            # Handle and visualize DBSCAN method-wise:
            df_no_outliers_dbscan = outlier_detector_dbscan.handle_outliers(method='remove', outlier_column='Outlier_DBSCAN')
            outlier_detector_dbscan.visualize_pca_with_colors(df=df, outlier_column='Outlier_DBSCAN', title="PCA - DBSCAN Outlier Detection")   
            # notes
            #the data is standardized (e.g., using StandardScaler) before applying DBSCAN.
            # This will scale the features to a common range, avoiding one feature from dominating the distance computation.'''
        pass               