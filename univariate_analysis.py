from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy import stats

# Define color palettes for visualizations
color_palette = {
    'primary': '#1f77b4',   # Blue for primary plots
    'secondary': '#ff7f0e', # Orange for secondary plots
    'tertiary': '#2ca02c',  # Green for tertiary plots
    'dark': '#333333',      # Dark Gray for titles and text
    'light': '#d62728',     # Red for highlighting
    'bar_vibrant': ['#ff6f61', '#6b5b95', '#88b04b', '#f7cac9', '#92a8d1']  # Vibrant colors for bar plots
}

# Abstract Base Class for Univariate Analysis Strategy
class UnivariateAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame, feature: str):
        """
        Perform univariate analysis on a specific feature of the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature (str): The name of the feature/column to be analyzed.

        Returns:
        None: This method visualizes the distribution of the feature.
        """
        pass

# Concrete Strategy for Advanced Numerical Features Analysis
class NumericalUnivariateAnalysis(UnivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature: str):
        """
        Perform advanced numerical analysis and visualization on a feature.
        Displays various plots including histogram with KDE, box plot, KDE plot,
        violin plot, cumulative frequency plot, frequency polygon, density plot,
        ECDF plot, and pair plot.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature (str): The name of the numerical feature/column to be analyzed.

        Returns:
        None: Executes various visualization methods.
        """
        print(f"Analysis for Numerical Feature '{feature}':")
        print("\nDataFrame Information:")
        df.info()
        print("\nDescriptive Statistics:")
        print(df[feature].describe())

        self._print_statistics(df, feature)
        self._plot_histogram_with_kde(df, feature)
        self._plot_boxplot(df, feature)
        self._plot_kde(df, feature)
        self._plot_violin(df, feature)
        self._plot_cumulative_frequency(df, feature)
        self._plot_frequency_polygon(df, feature)
        self._plot_density(df, feature)
        self._plot_ecdf(df, feature)
       
    def _print_statistics(self, df: pd.DataFrame, feature: str):
        """
        Print statistical measures for the numerical feature including mean,
        median, standard deviation, skewness, kurtosis, trimmed mean, and median 
        absolute deviation.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature (str): The name of the numerical feature/column to be analyzed.

        Returns:
        None: Outputs statistical measures to the console.
        """
        print(f"Mean: {df[feature].mean()}")
        print(f"Median: {df[feature].median()}")
        print(f"Standard Deviation: {df[feature].std()}")
        print(f"Skewness: {df[feature].skew()}")
        print(f"Kurtosis: {df[feature].kurtosis()}")
        print(f"Trimmed Mean (10%): {stats.trim_mean(df[feature], proportiontocut=0.1)}")
        print(f"Median Absolute Deviation: {stats.median_abs_deviation(df[feature])}")

    def _plot_histogram_with_kde(self, df: pd.DataFrame, feature: str):
        """
        Plot histogram with KDE (Kernel Density Estimate) to show the distribution of 
        the numerical feature.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature (str): The name of the numerical feature/column to be analyzed.

        Returns:
        None: Displays a histogram with KDE plot.
        """
        plt.figure(figsize=(10, 6))
        sns.histplot(df[feature], kde=True, bins=30, color=color_palette['primary'])
        plt.title(f'Histogram and KDE of {feature}', fontsize=18, color=color_palette['dark'])
        plt.xlabel(feature, fontsize=14)
        plt.ylabel('Density', fontsize=14)
        plt.grid(True)
        plt.show()

    def _plot_boxplot(self, df: pd.DataFrame, feature: str):
        """
        Plot a box plot to visualize the distribution and outliers of the numerical feature.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature (str): The name of the numerical feature/column to be analyzed.

        Returns:
        None: Displays a box plot.
        """
        fig = px.box(df, y=feature, title=f'Box Plot of {feature}', color_discrete_sequence=[color_palette['primary']])
        fig.update_layout(
            title_x=0.5,
            title_font_size=24,
            title_font_color=color_palette['dark'],
            yaxis_title=feature
        )
        fig.show()

    def _plot_kde(self, df: pd.DataFrame, feature: str):
        """
        Plot Kernel Density Estimate (KDE) to show the density distribution of the numerical feature.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature (str): The name of the numerical feature/column to be analyzed.

        Returns:
        None: Displays a KDE plot.
        """
        plt.figure(figsize=(10, 6))
        sns.kdeplot(df[feature], fill=True, color=color_palette['primary'])
        plt.title(f'Density Plot (KDE) of {feature}', fontsize=18, color=color_palette['dark'])
        plt.xlabel(feature, fontsize=14)
        plt.ylabel('Density', fontsize=14)
        plt.grid(True)
        plt.show()

    def _plot_violin(self, df: pd.DataFrame, feature: str):
        """
        Plot a violin plot to show the distribution of the numerical feature with density and box plot.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature (str): The name of the numerical feature/column to be analyzed.

        Returns:
        None: Displays a violin plot.
        """
        fig = px.violin(df, y=feature, box=True, points="all", title=f'Violin Plot of {feature}', color_discrete_sequence=[color_palette['primary']])
        fig.update_layout(
            title_x=0.5,
            title_font_size=24,
            title_font_color=color_palette['dark']
        )
        fig.show()

    def _plot_cumulative_frequency(self, df: pd.DataFrame, feature: str):
        """
        Plot a cumulative frequency plot for the numerical feature.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature (str): The name of the numerical feature/column to be analyzed.

        Returns:
        None: Displays a cumulative frequency plot.
        """
        sorted_data = np.sort(df[feature].dropna())
        cumulative_freq = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sorted_data, y=cumulative_freq, mode='markers+lines', marker=dict(color=color_palette['primary'])))
        fig.update_layout(
            title=f'Cumulative Frequency Plot of {feature}',
            xaxis_title=feature,
            yaxis_title='Cumulative Frequency',
            title_x=0.5,
            title_font_size=24,
            title_font_color=color_palette['dark']
        )
        fig.show()

    def _plot_frequency_polygon(self, df: pd.DataFrame, feature: str):
        """
        Plot a frequency polygon to show the distribution of the numerical feature.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature (str): The name of the numerical feature/column to be analyzed.

        Returns:
        None: Displays a frequency polygon plot.
        """
        hist, bins = np.histogram(df[feature].dropna(), bins=10)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=bin_centers, y=hist, mode='lines+markers', marker=dict(color=color_palette['primary'])))
        fig.update_layout(
            title=f'Frequency Polygon of {feature}',
            xaxis_title=feature,
            yaxis_title='Frequency',
            title_x=0.5,
            title_font_size=24,
            title_font_color=color_palette['dark']
        )
        fig.show()

    def _plot_density(self, df: pd.DataFrame, feature: str):
        """
        Plot a density plot to show the distribution of the numerical feature.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature (str): The name of the numerical feature/column to be analyzed.

        Returns:
        None: Displays a density plot.
        """
        plt.figure(figsize=(10, 6))
        sns.histplot(df[feature], kde=True, color=color_palette['primary'])
        plt.title(f'Density Plot of {feature}', fontsize=18, color=color_palette['dark'])
        plt.xlabel(feature, fontsize=14)
        plt.ylabel('Density', fontsize=14)
        plt.grid(True)
        plt.show()

    def _plot_ecdf(self, df: pd.DataFrame, feature: str):
        """
        Plot an Empirical Cumulative Distribution Function (ECDF) to show the proportion of observations 
        less than or equal to each value of the numerical feature.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature (str): The name of the numerical feature/column to be analyzed.

        Returns:
        None: Displays an ECDF plot.
        """
        sorted_data = np.sort(df[feature].dropna())
        ecdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        plt.figure(figsize=(10, 6))
        plt.step(sorted_data, ecdf, where='post', color=color_palette['primary'])
        plt.title(f'ECDF of {feature}', fontsize=18, color=color_palette['dark'])
        plt.xlabel(feature, fontsize=14)
        plt.ylabel('ECDF', fontsize=14)
        plt.grid(True)
        plt.show()

  

    
# Concrete Strategy for Categorical Features Analysis
class CategoricalUnivariateAnalysis(UnivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature: str):
        """
        Perform categorical feature analysis and visualization.
        Displays pie chart, exploded pie chart, donut chart, count plot, and frequency table.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature (str): The name of the categorical feature/column to be analyzed.

        Returns:
        None: Executes various visualization methods.
        """
        print(f"Analysis for Categorical Feature '{feature}':")
        print("\nDataFrame Information:")
        df.info()
        print("\nFrequency Table:")
        print(df[feature].value_counts())

        self._plot_barplot(df, feature)
        self._plot_pie_chart(df, feature)
        self._plot_exploded_pie_chart(df, feature)
        self._plot_donut_chart(df, feature)
        self._plot_countplot(df, feature)
        self._print_frequency_table(df, feature)
        self._plot_stripplot_with_counts_and_size(df, feature)
          
    def _plot_barplot(self, df: pd.DataFrame, feature: str):
          """
          Plot a bar plot to visualize the frequency of each category in the categorical feature.

          Parameters:
          df (pd.DataFrame): The dataframe containing the data.
          feature (str): The name of the categorical feature/column to be analyzed.

          Returns:
          None: Displays a bar plot.
          """
          plt.figure(figsize=(10, 6))
          
          # Get value counts and the corresponding index (categories)
          value_counts = df[feature].value_counts()
          categories = value_counts.index
          counts = value_counts.values

          # Adjust palette length to match the number of categories
          num_categories = len(categories)
          palette = color_palette['bar_vibrant'][:num_categories]

          # Create the bar plot
          sns.barplot(x=categories, y=counts, palette=palette, hue=categories)
          
          plt.title(f'Bar Plot of {feature}', fontsize=18, color=color_palette['dark'])
          plt.xlabel(feature, fontsize=14)
          plt.ylabel('Count', fontsize=14)
          plt.xticks(rotation=45)
          plt.grid(True)
          plt.legend(title=feature, loc='upper right', bbox_to_anchor=(1.1, 1), borderaxespad=0.)
          plt.show()


    def _plot_pie_chart(self, df: pd.DataFrame, feature: str):
        """
        Plot a pie chart to show the proportion of each category in the categorical feature.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature (str): The name of the categorical feature/column to be analyzed.

        Returns:
        None: Displays a pie chart.
        """
        counts = df[feature].value_counts()
        fig = px.pie(values=counts, names=counts.index, title=f'Pie Chart of {feature}', color_discrete_sequence=color_palette['bar_vibrant'])
        fig.update_layout(
            title_x=0.5,
            title_font_size=24,
            title_font_color=color_palette['dark']
        )
        fig.show()

    def _plot_exploded_pie_chart(self, df: pd.DataFrame, feature: str):
        """
        Plot an exploded pie chart to highlight specific categories.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature (str): The name of the categorical feature/column to be analyzed.

        Returns:
        None: Displays an exploded pie chart.
        """
        counts = df[feature].value_counts()
        fig = px.pie(values=counts, names=counts.index, title=f'Exploded Pie Chart of {feature}', 
                     hole=0.3, color_discrete_sequence=color_palette['bar_vibrant'])
        fig.update_traces(hoverinfo='label+percent', textinfo='percent+label', 
                          pull=[0.1 if i == 0 else 0 for i in range(len(counts))])
        fig.update_layout(
            title_x=0.5,
            title_font_size=24,
            title_font_color=color_palette['dark']
        )
        fig.show()

    def _plot_donut_chart(self, df: pd.DataFrame, feature: str):
        """
        Plot a donut chart to show the proportion of each category with a hole in the center.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature (str): The name of the categorical feature/column to be analyzed.

        Returns:
        None: Displays a donut chart.
        """
        counts = df[feature].value_counts()
        fig = px.pie(values=counts, names=counts.index, title=f'Donut Chart of {feature}', 
                     hole=0.4, color_discrete_sequence=color_palette['bar_vibrant'])
        fig.update_layout(
            title_x=0.5,
            title_font_size=24,
            title_font_color=color_palette['dark']
        )
        fig.show()

    def _plot_countplot(self, df: pd.DataFrame, feature: str):
        """
        Plot a count plot to visualize the frequency of each category in the categorical feature.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature (str): The name of the categorical feature/column to be analyzed.

        Returns:
        None: Displays a count plot.
        """
        plt.figure(figsize=(10, 6))
        sns.countplot(x=feature, data=df, palette=color_palette['bar_vibrant'])
        plt.title(f'Count Plot of {feature}', fontsize=18, color=color_palette['dark'])
        plt.xlabel(feature, fontsize=14)
        plt.ylabel('Count', fontsize=14)
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.show()

    def _print_frequency_table(self, df: pd.DataFrame, feature: str):
        """
        Print a table showing the frequency count of each category in the categorical feature.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature (str): The name of the categorical feature/column to be analyzed.

        Returns:
        None: Outputs the frequency table to the console.
        """
        print(df[feature].value_counts().sort_index())

    def _plot_stripplot_with_counts_and_size(self, df: pd.DataFrame, feature: str):
      """
      Plot a strip plot where y-values represent counts of the categorical feature,
      and the size of each point corresponds to the count of each category.

      Parameters:
      df (pd.DataFrame): The dataframe containing the data.
      feature (str): The name of the categorical feature/column to be analyzed.

      Returns:
      None: Displays a strip plot with counts as y-values and sizes corresponding to counts.
      """
      # Calculate counts for each category
      counts = df[feature].value_counts().reset_index()
      counts.columns = [feature, 'Count']
      
      plt.figure(figsize=(10, 6))

      # Create a strip plot using seaborn
      sns.stripplot(x=counts[feature], y=counts['Count'], jitter=True, hue=counts[feature], palette='viridis', dodge=True, legend=False)

     

      plt.title(f'Strip Plot of {feature} with Counts and Sizes', fontsize=18, color='darkblue')
      plt.xlabel(feature, fontsize=14)
      plt.ylabel('Count', fontsize=14)
      plt.grid(True)
      plt.show()

    

## Context Class that uses a UnivariateAnalysisStrategy
class UnivariateAnalyzer:
    def __init__(self, strategy: UnivariateAnalysisStrategy):
        """
        Initializes the UnivariateAnalyzer with a specific analysis strategy.

        Parameters:
        strategy (UnivariateAnalysisStrategy): The strategy to be used for univariate analysis.

        Returns:
        None
        """
        self._strategy = strategy

    def set_strategy(self, strategy: UnivariateAnalysisStrategy):
        """
        Sets a new strategy for the UnivariateAnalyzer.

        Parameters:
        strategy (UnivariateAnalysisStrategy): The new strategy to be used for univariate analysis.

        Returns:
        None
        """
        self._strategy = strategy

    def execute_analysis(self, df: pd.DataFrame, feature: str):
        """
        Executes the univariate analysis using the current strategy.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature (str): The name of the feature/column to be analyzed.

        Returns:
        None: Executes the strategy's analysis method and visualizes the results.
        """
        self._strategy.analyze(df, feature)


# Example usage of the UnivariateAnalyzer with different strategies.
if __name__ == "__main__":
    """# Create a sample dataframe for demonstration
    np.random.seed(0)
    df = pd.DataFrame({
        'numerical_feature': np.random.normal(loc=50, scale=10, size=1000),
        'categorical_feature': np.random.choice(['A', 'B', 'C', 'D'], size=1000)
    })

    # Analyzing a numerical feature
    print("Numerical Feature Analysis:")
    analyzer = UnivariateAnalyzer(NumericalUnivariateAnalysis())
    analyzer.execute_analysis(df, 'numerical_feature')

    # Analyzing a categorical feature
    print("\nCategorical Feature Analysis:")
    analyzer.set_strategy(CategoricalUnivariateAnalysis())
    analyzer.execute_analysis(df, 'categorical_feature')"""
    pass

