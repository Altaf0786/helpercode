from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import missingno as msno

# Abstract Base Class for Missing Values Analysis
# -----------------------------------------------
class MissingValuesAnalysisTemplate(ABC):
    def analyze(self, df: pd.DataFrame):
        """
        Performs a complete missing values analysis by identifying and visualizing missing values.

        Parameters:
        df (pd.DataFrame): The dataframe to be analyzed.

        Returns:
        None: This method performs the analysis and visualizes missing values.
        """
        self.identify_missing_values(df)
        self.visualize_missing_values(df)

    @abstractmethod
    def identify_missing_values(self, df: pd.DataFrame):
        """
        Identifies missing values in the dataframe and prints the count.

        Parameters:
        df (pd.DataFrame): The dataframe to be analyzed.

        Returns:
        None: Prints missing values count and percentage.
        """
        pass

    @abstractmethod
    def visualize_missing_values(self, df: pd.DataFrame):
        """
        Visualizes missing values in the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe to be visualized.

        Returns:
        None: Creates visualizations (e.g., matrix, bar, heatmap, dendrogram).
        """
        pass


# Concrete Class for Full Missingno Analysis
# -------------------------------------------
class FullMissingnoAnalysis(MissingValuesAnalysisTemplate):
    def identify_missing_values(self, df: pd.DataFrame):
        """
        Identifies and prints missing values in the dataframe, sorted by percentage in ascending order.

        Parameters:
        df (pd.DataFrame): The dataframe to be analyzed.

        Returns:
        None: Prints missing values summary in ascending order of percentage.
        
        """
        # Calculate missing values and percentage
        missing_summary = df.isnull().sum()
        missing_percentage = (missing_summary / len(df)) * 100
        summary_df = pd.DataFrame({
            'Missing Values': missing_summary,
            'Percentage': missing_percentage
        }).sort_values(by='Percentage', ascending=True)
        
        # Filter out columns with no missing values
        summary_df = summary_df[summary_df['Missing Values'] > 0]
        
        # Print summary
        print("\nMissing Values Summary (Sorted by Percentage):")
        print(summary_df)

    def visualize_missing_values(self, df: pd.DataFrame):
        """
        Creates multiple missing values visualizations using the missingno library.

        Parameters:
        df (pd.DataFrame): The dataframe to be visualized.

        Returns:
        None: Displays missingno visualizations (matrix, bar, heatmap, dendrogram).
        """
        print("\nVisualizing Missing Values with Missingno...")

        # Matrix plot
        plt.figure()
        msno.matrix(df)
        plt.title("Missing Values Matrix")
        plt.show()

        # Bar plot
        plt.figure()
        msno.bar(df)
        plt.title("Missing Values Bar Plot")
        plt.show()

        # Heatmap
        plt.figure()
        msno.heatmap(df)
        plt.title("Missing Values Heatmap")
        plt.show()

        # Dendrogram
        plt.figure()
        msno.dendrogram(df)
        plt.title("Missing Values Dendrogram")
        plt.show()


# Example usage
if __name__ == "__main__":
    '''# Load the data (Example using Titanic dataset)
    url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
    df = pd.read_csv(url)

    # Perform Full Missingno Missing Values Analysis
    print("Using FullMissingnoAnalysis:")
    full_missingno_analyzer = FullMissingnoAnalysis()
    full_missingno_analyzer.analyze(df)'''
pass