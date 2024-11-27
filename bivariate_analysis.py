from abc import ABC, abstractmethod
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.graphics.mosaicplot import mosaic

# Abstract Base Class for Bivariate Analysis
class BivariateAnalysisTemplate(ABC):
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str, plots: list = None, hue: str = None, **kwargs):
        """
        Perform bivariate analysis between two features with selected plots and optional hue.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data to be analyzed.
        feature1 (str): The name of the first feature.
        feature2 (str): The name of the second feature.
        plots (list): List of plots to generate (e.g., ['scatter', 'line']).
        hue (str): Feature for grouping data by color (optional).
        kwargs: Additional parameters for visualization.

        Returns:
        None
        """
        if plots is None:
            plots = self.default_plots()
        for plot_type in plots:
            self.plot(df, feature1, feature2, plot_type, hue, **kwargs)

    @abstractmethod
    def plot(self, df: pd.DataFrame, feature1: str, feature2: str, plot_type: str, hue: str, **kwargs):
        pass

    @abstractmethod
    def default_plots(self):
        pass


# Numerical vs Numerical Analysis
class NumericalVsNumericalAnalysis(BivariateAnalysisTemplate):
    def plot(self, df: pd.DataFrame, feature1: str, feature2: str, plot_type: str, hue: str, **kwargs):
        if plot_type == "scatter":
            sns.scatterplot(data=df, x=feature1, y=feature2, hue=hue, **kwargs)
            plt.title(f"Scatter Plot: {feature1} vs {feature2} (Hue: {hue})")
        elif plot_type == "regression":
            sns.lmplot(data=df, x=feature1, y=feature2, hue=hue, **kwargs)
            plt.title(f"Regression Plot: {feature1} vs {feature2} (Hue: {hue})")
            return
        elif plot_type == "hexbin":
            sns.jointplot(data=df, x=feature1, y=feature2, kind="hex", hue=hue, **kwargs)
            plt.suptitle(f"Hexbin Plot: {feature1} vs {feature2} (Hue: {hue})", y=1.02)
            return
        elif plot_type == "line":
            sns.lineplot(data=df, x=feature1, y=feature2, hue=hue, **kwargs)
            plt.title(f"Line Plot: {feature1} vs {feature2} (Hue: {hue})")
        elif plot_type == "kde":
            sns.kdeplot(data=df, x=feature1, y=feature2, hue=hue, cmap="coolwarm", **kwargs)
            plt.title(f"KDE Plot: {feature1} vs {feature2} (Hue: {hue})")
        else:
            raise ValueError(f"Unsupported plot type: {plot_type}")
        plt.show()

    def default_plots(self):
        return ["scatter", "regression", "hexbin", "line", "kde"]


# Numerical vs Categorical Analysis
class NumericalVsCategoricalAnalysis(BivariateAnalysisTemplate):
    def plot(self, df: pd.DataFrame, feature1: str, feature2: str, plot_type: str, hue: str, **kwargs):
        if plot_type == "box":
            sns.boxplot(data=df, x=feature2, y=feature1, hue=hue, **kwargs)
            plt.title(f"Box Plot: {feature1} by {feature2} (Hue: {hue})")
        elif plot_type == "strip":
            sns.stripplot(data=df, x=feature2, y=feature1, hue=hue, **kwargs)
            plt.title(f"Strip Plot: {feature1} by {feature2} (Hue: {hue})")
        elif plot_type == "point":
            sns.pointplot(data=df, x=feature2, y=feature1, hue=hue, **kwargs)
            plt.title(f"Point Plot: {feature1} by {feature2} (Hue: {hue})")
        elif plot_type == "violin":
            sns.violinplot(data=df, x=feature2, y=feature1, hue=hue, split=True, **kwargs)
            plt.title(f"Violin Plot: {feature1} by {feature2} (Hue: {hue})")
        elif plot_type == "swarm":
            sns.swarmplot(data=df, x=feature2, y=feature1, hue=hue, **kwargs)
            plt.title(f"Swarm Plot: {feature1} by {feature2} (Hue: {hue})")
        else:
            raise ValueError(f"Unsupported plot type: {plot_type}")
        plt.show()

    def default_plots(self):
        return ["box", "strip", "point", "violin", "swarm"]


# Categorical vs Categorical Analysis
class CategoricalVsCategoricalAnalysis(BivariateAnalysisTemplate):
    def plot(self, df: pd.DataFrame, feature1: str, feature2: str, plot_type: str, hue: str, **kwargs):
        if plot_type == "count":
            sns.countplot(data=df, x=feature1, hue=feature2 if hue is None else hue, **kwargs)
            plt.title(f"Count Plot: {feature1} by {feature2} (Hue: {hue})")
        elif plot_type == "heatmap":
            cross_tab = pd.crosstab(df[feature1], df[feature2])
            sns.heatmap(cross_tab, annot=True, fmt="d", cmap="coolwarm", **kwargs)
            plt.title(f"Heatmap: {feature1} vs {feature2}")
        elif plot_type == "mosaic":
            mosaic(df, [feature1, feature2])
            plt.title(f"Mosaic Plot: {feature1} vs {feature2}")
        else:
            raise ValueError(f"Unsupported plot type: {plot_type}")
        plt.show()

    def default_plots(self):
        return ["count", "heatmap", "mosaic"]


# Example Usage
if __name__ == "__main__":
   """ # Load example dataset
    df = sns.load_dataset("tips")

    # Numerical vs Numerical Analysis
    num_vs_num = NumericalVsNumericalAnalysis()
    num_vs_num.analyze(df, "total_bill", "tip", plots=["scatter", "regression", "kde"], hue="sex")

    # Numerical vs Categorical Analysis
    num_vs_cat = NumericalVsCategoricalAnalysis()
    num_vs_cat.analyze(df, "total_bill", "day", plots=["box", "violin"], hue="sex")

    # Categorical vs Categorical Analysis
    cat_vs_cat = CategoricalVsCategoricalAnalysis()
    cat_vs_cat.analyze(df, "sex", "smoker", plots=["count", "heatmap"])"""
pass    
