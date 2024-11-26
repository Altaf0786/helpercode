from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import chi2_contingency, ttest_ind, f_oneway, spearmanr, kendalltau
import statsmodels.api as sm

# Abstract Base Class for Bivariate Analysis Strategy
# Abstract Base Class for Bivariate Analysis Strategy
class BivariateAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str, hue: str = None, **kwargs):
        """
        Perform bivariate analysis on two features of the dataframe. This method can be used to 
        visualize and/or compute statistics about the relationship between two features.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature1 (str): The name of the first feature/column to be analyzed.
        feature2 (str): The name of the second feature/column to be analyzed.
        hue (str): The name of the categorical feature/column to be used as hue (optional).
        **kwargs: Additional parameters to modify or control the analysis behavior.

        Returns:
        None: This method is responsible for visualizing or analyzing the relationship between the two features.
        
        Example:
        You could pass additional parameters to control plot styling, statistical test options, etc.
        """
        pass

class ContinuousVsContinuousAnalysis(BivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str, hue: str = None, plots=None,**kwargs):
        # Define a dictionary of plot methods
        plot_methods = {
            'scatter': self._plot_scatter,
            'regression': self._plot_regression,
            'residuals': self._plot_residuals,
            'hexbin': self._plot_hexbin,
            'kde': self._plot_kde,
            'pairplot': self._plot_pairplot,
            'bubble': self._plot_bubble,
            'correlation_heatmap': self._plot_correlation_heatmap
        }

        # If no specific plots are requested, plot all
        if plots is None:
            plots = plot_methods.keys()

        # Call the specified plot methods
        for plot in plots:
            if plot in plot_methods:
                if plot == 'pairplot':  # Handle pairplot separately as it requires a list of features
                    plot_methods[plot](df, [feature1, feature2])
                else:
                    plot_methods[plot](df, feature1, feature2, hue)

    def _plot_scatter(self, df: pd.DataFrame, feature1: str, feature2: str, hue: str = None):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=feature1, y=feature2, hue=hue, data=df)
        plt.title(f'Scatter Plot of {feature1} vs {feature2}')
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.show()
    



    def _plot_regression(self, df: pd.DataFrame, feature1: str, feature2: str):
        plt.figure(figsize=(10, 6))
        sns.regplot(x=feature1, y=feature2, data=df)
        plt.title(f'Regression Line of {feature1} vs {feature2}')
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.show()

    def _plot_residuals(self, df: pd.DataFrame, feature1: str, feature2: str):
        # Clean the data by dropping rows with missing or infinite values
        df_clean = df[[feature1, feature2]].replace([np.inf, -np.inf], np.nan).dropna()
        
        if df_clean.empty:
            print("No data available for residual plot after cleaning.")
            return  # Exits the function if no data is available

        # Fit a linear model
        X = sm.add_constant(df_clean[feature1])  # Adds a constant term to the predictor
        model = sm.OLS(df_clean[feature2], X).fit()
        residuals = df_clean[feature2] - model.predict(X)
        
        # Plot residuals
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=df_clean[feature1], y=residuals)
        plt.title('Residual Plot')
        plt.xlabel(feature1)
        plt.ylabel('Residuals')
        plt.axhline(0, color='red', linestyle='--')  # Adds a horizontal line at 0
        plt.show()


    def _plot_hexbin(self, df: pd.DataFrame, feature1: str, feature2: str):
        plt.figure(figsize=(10, 6))
        plt.hexbin(df[feature1], df[feature2], gridsize=30, cmap='Blues')
        plt.colorbar(label='Count')
        plt.title(f'Hexbin Plot of {feature1} vs {feature2}')
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.show()

    def _plot_kde(self, df: pd.DataFrame, feature1: str, feature2: str):
        if df[feature1].var() == 0 or df[feature2].var() == 0:
            print(f"Warning: One or both features '{feature1}' and '{feature2}' have zero variance. KDE cannot be computed.")
            return

        plt.figure(figsize=(10, 6))
        sns.kdeplot(x=df[feature1], y=df[feature2], cmap='Blues', fill=True)
        plt.title(f'KDE Plot of {feature1} vs {feature2}')
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.show()

    def _plot_pairplot(self, df: pd.DataFrame, features: list):
        plt.figure(figsize=(10, 6))
        sns.pairplot(df[features])
        plt.title(f'Pair Plot of {", ".join(features)}')
        plt.show()

    def _plot_bubble(self, df: pd.DataFrame, feature1: str, feature2: str):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=feature1, y=feature2, size=df[feature2], data=df, sizes=(20, 200))
        plt.title(f'Bubble Plot of {feature1} vs {feature2}')
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.show()

    def _plot_correlation_heatmap(self, df: pd.DataFrame, features: list):
        correlation_matrix = df[features].corr()
        plt.figure(figsize=(10, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title(f'Correlation Heatmap')
        plt.show()


# Concrete Strategy for Continuous vs Categorical

class ContinuousVsCategoricalAnalysis(BivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, continuous_feature: str, categorical_feature: str, hue: str = None, top_n_categories: int = 10, plots=None):
        # Filter for the top N categories based on frequency
        df_filtered = self._filter_high_cardinality(df, categorical_feature, top_n_categories)

        # Define a dictionary of plot methods
        plot_methods = {
            'boxplot': self._plot_boxplot,
            'violin': self._plot_violin,
            'bar_with_error': self._plot_bar_with_error,
            'strip': self._plot_strip,
            'swarm': self._plot_swarm,
            'boxen': self._plot_boxen,
            'point': self._plot_point,
            'ecdf': self._plot_ecdf,
            'barplot': self._barplot,
            'heatmap_by_feature' :self._plot_heatmap_by_feature
        }

        # If no specific plots are requested, plot all
        if plots is None:
            plots = plot_methods.keys()

        # Call the specified plot methods
        for plot in plots:
            if plot in plot_methods:
                plot_methods[plot](df_filtered, continuous_feature, categorical_feature, hue)

    def _filter_high_cardinality(self, df: pd.DataFrame, categorical_feature: str, top_n_categories: int):
        # Select only top N most frequent categories or group the rest into 'Other'
        top_categories = df[categorical_feature].value_counts().nlargest(top_n_categories).index
        df_filtered = df[df[categorical_feature].isin(top_categories)]
        return df_filtered
    
    def _plot_heatmap_by_feature(df: pd.DataFrame, categorical_feature: str, continuous_feature: str, extractor_function=None, figsize=(5, 25)):
        """
        Plots a heatmap of the average continuous feature per categorical feature, sorted by a specified extractor function.

        Parameters:
        - df (pd.DataFrame): The input dataframe containing the data.
        - categorical_feature (str): The column name that holds the categorical feature (e.g., sector).
        - continuous_feature (str): The column name that holds the continuous feature (e.g., price).
        - extractor_function (function, optional): A function to extract a sorting key (e.g., for sorting sectors by number).
        - figsize (tuple): The size of the plot (default is (5, 25)).

        Returns:
        - None
        """
        # Group by the categorical feature and calculate the average of the continuous feature
        avg_feature_per_category = df.groupby(categorical_feature)[continuous_feature].mean().reset_index()

        # Apply extractor function for sorting if provided
        if extractor_function:
            avg_feature_per_category['sort_key'] = avg_feature_per_category[categorical_feature].apply(extractor_function)
            avg_feature_per_category_sorted = avg_feature_per_category.sort_values(by='sort_key')
        else:
            avg_feature_per_category_sorted = avg_feature_per_category.sort_values(by=categorical_feature)

        # Plot the heatmap
        plt.figure(figsize=figsize)
        sns.heatmap(avg_feature_per_category_sorted.set_index(categorical_feature)[[continuous_feature]], annot=True, fmt=".2f", linewidths=.5)
        plt.title(f'Average {continuous_feature} per {categorical_feature}')
        plt.xlabel(f'Average {continuous_feature}')
        plt.ylabel(categorical_feature)
        plt.show()

    
    def _plot_boxplot(self, df: pd.DataFrame, continuous_feature: str, categorical_feature: str, hue: str = None):
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=categorical_feature, y=continuous_feature, hue=hue, data=df)
        plt.title(f'Box Plot of {continuous_feature} by {categorical_feature}')
        plt.xlabel(categorical_feature)
        plt.ylabel(continuous_feature)
        plt.xticks(rotation=45)
        plt.show()

    def _barplot(self, df: pd.DataFrame, continuous_feature: str, categorical_feature: str, hue: str = None, estimator=None):
        plt.figure(figsize=(10, 6)) 
        sns.barplot(x=categorical_feature, y=continuous_feature, hue=hue, data=df, estimator=estimator)
        plt.title(f'Barplot of {continuous_feature} by {categorical_feature}')
        plt.xlabel(categorical_feature)
        plt.ylabel(continuous_feature)
        plt.show()

    def _plot_violin(self, df: pd.DataFrame, continuous_feature: str, categorical_feature: str, hue: str = None):
        plt.figure(figsize=(10, 6))
        sns.violinplot(x=categorical_feature, y=continuous_feature, hue=hue, data=df)
        plt.title(f'Violin Plot of {continuous_feature} by {categorical_feature}')
        plt.xlabel(categorical_feature)
        plt.ylabel(continuous_feature)
        plt.show()

    def _plot_bar_with_error(self, df: pd.DataFrame, continuous_feature: str, categorical_feature: str):
        df_grouped = df.groupby(categorical_feature)[continuous_feature].agg(['mean', 'std'])
        df_grouped.plot(kind='bar', y='mean', yerr='std', capsize=4)
        plt.title(f'Bar Plot with Error Bars for {continuous_feature} by {categorical_feature}')
        plt.xlabel(categorical_feature)
        plt.ylabel(continuous_feature)
        plt.show()

    def _plot_strip(self, df: pd.DataFrame, continuous_feature: str, categorical_feature: str, hue: str = None):
        plt.figure(figsize=(10, 6))
        sns.stripplot(x=categorical_feature, y=continuous_feature, hue=hue, data=df, jitter=True)
        plt.title(f'Strip Plot of {continuous_feature} by {categorical_feature}')
        plt.xlabel(categorical_feature)
        plt.ylabel(continuous_feature)
        plt.show()

    def _plot_swarm(self, df: pd.DataFrame, continuous_feature: str, categorical_feature: str, hue: str = None):
        plt.figure(figsize=(10, 6))
        sns.swarmplot(x=categorical_feature, y=continuous_feature, hue=hue, data=df)
        plt.title(f'Swarm Plot of {continuous_feature} by {categorical_feature}')
        plt.xlabel(categorical_feature)
        plt.ylabel(continuous_feature)
        plt.show()

    def _plot_boxen(self, df: pd.DataFrame, continuous_feature: str, categorical_feature: str):
        plt.figure(figsize=(10, 6))
        sns.boxenplot(x=categorical_feature, y=continuous_feature, data=df)
        plt.title(f'Boxen Plot of {continuous_feature} by {categorical_feature}')
        plt.xlabel(categorical_feature)
        plt.ylabel(continuous_feature)
        plt.show()

    def _plot_point(self, df: pd.DataFrame, continuous_feature: str, categorical_feature: str):
        plt.figure(figsize=(10, 6))
        sns.pointplot(x=categorical_feature, y=continuous_feature, data=df)
        plt.title(f'Point Plot of {continuous_feature} by {categorical_feature}')
        plt.xlabel(categorical_feature)
        plt.ylabel(continuous_feature)
        plt.show()

    def _plot_ecdf(self, df: pd.DataFrame, continuous_feature: str, categorical_feature: str):
        for category in df[categorical_feature].unique():
            subset = df[df[categorical_feature] == category]
            ecdf = subset[continuous_feature].rank() / len(subset[continuous_feature])
            plt.step(subset[continuous_feature].sort_values(), ecdf.sort_values(), where='post', label=str(category))
        plt.title(f'ECDF of {continuous_feature} by {categorical_feature}')
        plt.xlabel(continuous_feature)
        plt.ylabel('ECDF')
        plt.legend(title=categorical_feature) 
        plt.show()


# Concrete Strategy for Categorical vs Categorical
class CategoricalVsCategoricalAnalysis(BivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, categorical_feature1: str, categorical_feature2: str, hue: str = None,aggfunc: str = 'count', plots=None):
        # Define a dictionary of plot methods
        plot_methods = {
            'countplot': self._plot_countplot,
            'heatmap': self._plot_heatmap,
            'mosaic': self._plot_mosaic,
            'crosstab': self._plot_crosstab,
            'stacked_bar': self._plot_stacked_bar,
            'vp_with_categorical_data': self._plot_vp_with_categorical_data,
            'scatter_matrix': self._plot_scatter_matrix,
            'correlation_heatmap': self._plot_pivot_heatmap
        }

        # If no specific plots are requested, plot all
        if plots is None:
            plots = plot_methods.keys()

        # Call the specified plot methods
        for plot in plots:
            if plot in plot_methods:
                plot_methods[plot](df, categorical_feature1, categorical_feature2)
    
    def _plot_pivot_heatmap(self, df: pd.DataFrame, categorical_feature1: str, categorical_feature2: str, target: str, aggfunc: str='mean'):
        """
        Generalized function to plot a heatmap for two categorical features against a numerical target.
        
        Parameters:
        df (pd.DataFrame): The input dataframe.
        categorical_feature1 (str): The first categorical feature.
        categorical_feature2 (str): The second categorical feature.
        target (str): The numeric target variable to aggregate (e.g., 'price').
        aggfunc (str): The aggregation function, default is 'mean'.
        
        """
        # Create a pivot table using the two categorical features and the target variable
        pivot_table = pd.pivot_table(df, index=categorical_feature1, columns=categorical_feature2, values=target, aggfunc=aggfunc)
        
        # Set up the plot size
        plt.figure(figsize=(15, 4))
        
        # Create a heatmap with annotations
        sns.heatmap(pivot_table, annot=True, cmap="YlGnBu", fmt='.2f')  # You can change the colormap to your preference
        
        # Set title and display the plot
        plt.title(f'Heatmap of {target} by {categorical_feature1} and {categorical_feature2}')
        plt.show()

            
    def _plot_countplot(self, df: pd.DataFrame, categorical_feature1: str, categorical_feature2: str):
        plt.figure(figsize=(10, 6))
        sns.countplot(x=categorical_feature1, hue=categorical_feature2, data=df)
        plt.title(f'Count Plot of {categorical_feature1} and {categorical_feature2}')
        plt.xlabel(categorical_feature1)
        plt.ylabel('Count')
        plt.show()
        

    def _plot_heatmap(self, df: pd.DataFrame, categorical_feature1: str, categorical_feature2: str):
        crosstab = pd.crosstab(df[categorical_feature1], df[categorical_feature2])
        plt.figure(figsize=(10, 6))
        sns.heatmap(crosstab, annot=True, cmap='YlGnBu', fmt='d')
        plt.title(f'Heatmap of {categorical_feature1} vs {categorical_feature2}')
        plt.xlabel(categorical_feature2)
        plt.ylabel(categorical_feature1)
        plt.show()

    def _plot_mosaic(self, df: pd.DataFrame, categorical_feature1: str, categorical_feature2: str):
        from statsmodels.graphics.mosaicplot import mosaic
        plt.figure(figsize=(10, 6))
        mosaic(df, [categorical_feature1, categorical_feature2])
        plt.title(f'Mosaic Plot of {categorical_feature1} and {categorical_feature2}')
        plt.show()

    def _plot_crosstab(self, df: pd.DataFrame, categorical_feature1: str, categorical_feature2: str):
        crosstab = pd.crosstab(df[categorical_feature1], df[categorical_feature2])
        print(crosstab)

    def _plot_stacked_bar(self, df: pd.DataFrame, categorical_feature1: str, categorical_feature2: str):
        crosstab = pd.crosstab(df[categorical_feature1], df[categorical_feature2])
        crosstab.div(crosstab.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
        plt.title(f'Stacked Bar Plot of {categorical_feature1} vs {categorical_feature2}')
        plt.xlabel(categorical_feature1)
        plt.ylabel('Proportion')
        plt.show()

    def _plot_vp_with_categorical_data(self, df: pd.DataFrame, categorical_feature1: str, categorical_feature2: str):
        plt.figure(figsize=(10, 6))
        sns.violinplot(x=categorical_feature1, y=categorical_feature2, data=df)
        plt.title(f'Violin Plot of {categorical_feature1} vs {categorical_feature2}')
        plt.xlabel(categorical_feature1)
        plt.ylabel(categorical_feature2)
        plt.show()

    def _plot_scatter_matrix(self, df: pd.DataFrame, features: list):
        plt.figure(figsize=(10, 6))
        sns.pairplot(df[features], hue=features[0])
        plt.title(f'Scatter Matrix of {", ".join(features)}')
        plt.show()

# Context Class for Bivariate Analysis
class BivariateAnalysisContext:
    def __init__(self, strategy: BivariateAnalysisStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: BivariateAnalysisStrategy):
        self._strategy = strategy

    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str, hue: str = None,aggregate=None):
        self._strategy.analyze(df, feature1, feature2, hue,aggregate)
if __name__ == "__main__":
# Example usage
   ''' df = sns.load_dataset('titanic')  # Ensure the dataset contains 'age', 'fare', 'pclass', and 'survived'

    # Continuous vs Continuous
    context = BivariateAnalysisContext(ContinuousVsContinuousAnalysis())
    context.analyze(df, 'age', 'fare')

    # Continuous vs Categorical
    context.set_strategy(ContinuousVsCategoricalAnalysis())
    context.analyze(df, 'age', 'pclass',hue='pclass')

    # Categorical vs Categorical
    context.set_strategy(CategoricalVsCategoricalAnalysis())
    context.analyze(df, 'pclass', 'survived')
'''
pass