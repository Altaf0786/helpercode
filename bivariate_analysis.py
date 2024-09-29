from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency, ttest_ind, f_oneway, spearmanr, kendalltau
import statsmodels.api as sm

# Abstract Base Class for Bivariate Analysis Strategy
class BivariateAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str, hue: str = None):
        """
        Perform bivariate analysis on two features of the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature1 (str): The name of the first feature/column to be analyzed.
        feature2 (str): The name of the second feature/column to be analyzed.
        hue (str): The name of the categorical feature/column to be used as hue (optional).

        Returns:
        None: This method visualizes or analyzes the relationship between the two features.
        """
        pass

class ContinuousVsContinuousAnalysis(BivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str, hue: str = None):
        self._plot_scatter(df, feature1, feature2, hue)
        self._plot_regression(df, feature1, feature2)
        self._plot_residuals(df, feature1, feature2)
        self._plot_hexbin(df, feature1, feature2)
        self._plot_kde(df, feature1, feature2)
        self._plot_pairplot(df, [feature1, feature2])
        self._plot_bubble(df, feature1, feature2)
        self._plot_correlation_heatmap(df, [feature1, feature2])

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
            return

        X = sm.add_constant(df_clean[feature1])
        model = sm.OLS(df_clean[feature2], X).fit()
        residuals = df_clean[feature2] - model.predict(X)
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=df_clean[feature1], y=residuals)
        plt.title('Residual Plot')
        plt.xlabel(feature1)
        plt.ylabel('Residuals')
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
    def analyze(self, df: pd.DataFrame, continuous_feature: str, categorical_feature: str, hue: str = None):
        self._plot_boxplot(df, continuous_feature, categorical_feature, hue)
        self._plot_violin(df, continuous_feature, categorical_feature, hue)
        self._plot_bar_with_error(df, continuous_feature, categorical_feature)
        self._plot_strip(df, continuous_feature, categorical_feature, hue)
        self._plot_swarm(df, continuous_feature, categorical_feature, hue)
        self._plot_boxen(df, continuous_feature, categorical_feature)
        self._plot_point(df, continuous_feature, categorical_feature)
        self._plot_ecdf(df, continuous_feature, categorical_feature)
        self._barplot(df, continuous_feature, categorical_feature, hue,estimator=np.mean)
    def _plot_boxplot(self, df: pd.DataFrame, continuous_feature: str, categorical_feature: str, hue: str = None):
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=categorical_feature, y=continuous_feature, hue=hue, data=df)
        plt.title(f'Box Plot of {continuous_feature} by {categorical_feature}')
        plt.xlabel(categorical_feature)
        plt.ylabel(continuous_feature)
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
    def analyze(self, df: pd.DataFrame, categorical_feature1: str, categorical_feature2: str, hue: str = None):
        self._plot_countplot(df, categorical_feature1, categorical_feature2)
        self._plot_heatmap(df, categorical_feature1, categorical_feature2)
        self._plot_mosaic(df, categorical_feature1, categorical_feature2)
        self._plot_crosstab(df, categorical_feature1, categorical_feature2)
        self._plot_stacked_bar(df, categorical_feature1, categorical_feature2)
        self._plot_vp_with_categorical_data(df, categorical_feature1, categorical_feature2)
        self._plot_scatter_matrix(df, [categorical_feature1, categorical_feature2])

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

    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str, hue: str = None):
        self._strategy.analyze(df, feature1, feature2, hue)

# Example usage
df = sns.load_dataset('titanic')  # Ensure the dataset contains 'age', 'fare', 'pclass', and 'survived'

# Continuous vs Continuous
context = BivariateAnalysisContext(ContinuousVsContinuousAnalysis())
context.analyze(df, 'age', 'fare')

# Continuous vs Categorical
context.set_strategy(ContinuousVsCategoricalAnalysis())
context.analyze(df, 'age', 'pclass',hue='pclass')

# Categorical vs Categorical
context.set_strategy(CategoricalVsCategoricalAnalysis())
context.analyze(df, 'pclass', 'survived')
