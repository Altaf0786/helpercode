from abc import ABC, abstractmethod
import scipy.stats as stats
import numpy as np
import pandas as pd

# Abstract Base Class for Statistical Tests
class StatisticalTest(ABC):
    
    @abstractmethod
    def run_test(self, data, *cols, alpha=0.05):
        pass

# Chi-Square Test Implementation
class ChiSquareTest(StatisticalTest):
    
    def run_test(self, data, col1, col2, alpha=0.05):
        contingency_table = pd.crosstab(data[col1], data[col2])
        chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
        print(f"Chi-Square Test Statistic: {chi2}, p-value: {p}")
        if p < alpha:
            print("Reject the null hypothesis: The distributions of the two groups are significantly different.")
        else:
            print("Fail to reject the null hypothesis: The distributions of the two groups are not significantly different.")

        return chi2, p

        

# Mann-Whitney U Test Implementation
class MannWhitneyUTest(StatisticalTest):

    def run_test(self, data, col1, col2, alpha=0.05):
        """Performs the Mann-Whitney U test for two independent samples."""
        u_stat, p_val = stats.mannwhitneyu(data[col1].dropna(), data[col2].dropna())
        print(f"Mann-Whitney U Statistic: {u_stat}, p-value: {p_val}")
        if p_val < alpha:
            print("Reject the null hypothesis: The distributions of the two groups are significantly different.")
        else:
            print("Fail to reject the null hypothesis: The distributions of the two groups are not significantly different.")
        return u_stat, p_val

# Kruskal-Wallis Test Implementation
class KruskalTest(StatisticalTest):

    def run_test(self, data, *cols, alpha=0.05):
        """Performs Kruskal-Wallis H-test for independent samples."""
        groups = [data[col].dropna() for col in cols]
        h_stat, p_val = stats.kruskal(*groups)
        print(f"Kruskal-Wallis Statistic: {h_stat}, p-value: {p_val}")
        if p_val < alpha:
            print("Reject the null hypothesis: The distributions of the groups are significantly different.")
        else:
            print("Fail to reject the null hypothesis: The distributions of the groups are not significantly different.")
        return h_stat, p_val

# Correlation Test Implementation (Pearson, Spearman)
class CorrelationTest(StatisticalTest):

    def run_test(self, data, col1, col2, method='pearson', alpha=0.05):
        """Performs correlation tests between two variables."""
        if method == 'pearson':
            corr_stat, p_val = stats.pearsonr(data[col1].dropna(), data[col2].dropna())
            print(f"Pearson Correlation Coefficient: {corr_stat}, p-value: {p_val}")
        elif method == 'spearman':
            corr_stat, p_val = stats.spearmanr(data[col1].dropna(), data[col2].dropna())
            print(f"Spearman Correlation Coefficient: {corr_stat}, p-value: {p_val}")
        else:
            raise ValueError("Unsupported method. Choose either 'pearson' or 'spearman'.")
        
        if p_val < alpha:
            print("Reject the null hypothesis: Significant correlation exists.")
        else:
            print("Fail to reject the null hypothesis: No significant correlation.")
        return corr_stat, p_val
# t test implementation         
class TTest(StatisticalTest):
   def run_test(self,data, col1, col2, alpha=0.05):
        """Performs an independent two-sample t-test."""
        group1 = data[col1].dropna()
        group2 = data[col2].dropna()
        t_stat, p_val = stats.ttest_ind(group1, group2)
        print(f"t-test Statistic: {t_stat}, p-value: {p_val}")
        if p_val < alpha:
            print("Reject the null hypothesis: The means of the two groups are significantly different.")
        else:
            print("Fail to reject the null hypothesis: The means of the two groups are not significantly different.")
        return t_stat, p_val

# Two-Tailed T-Test Implementation
class TwoTailedTTest(StatisticalTest):

    def run_test(self, data, col1, col2, alpha=0.05):
        """Performs a two-tailed independent t-test."""
        group1 = data[col1].dropna()
        group2 = data[col2].dropna()
        t_stat, p_val = stats.ttest_ind(group1, group2)
        print(f"Two-Tailed t-test Statistic: {t_stat}, p-value: {p_val}")
        if p_val < alpha:
            print("Reject the null hypothesis: The means of the two groups are significantly different.")
        else:
            print("Fail to reject the null hypothesis: The means of the two groups are not significantly different.")
        return t_stat, p_val

# Paired T-Test Implementation
class PairedTTest(StatisticalTest):

    def run_test(self, data, col1, col2, alpha=0.05):
        """Performs a paired t-test for two dependent samples."""
        group1 = data[col1].dropna()
        group2 = data[col2].dropna()
        t_stat, p_val = stats.ttest_rel(group1, group2)
        print(f"Paired t-test Statistic: {t_stat}, p-value: {p_val}")
        if p_val < alpha:
            print("Reject the null hypothesis: The means of the paired groups are significantly different.")
        else:
            print("Fail to reject the null hypothesis: The means of the paired groups are not significantly different.")
        return t_stat, p_val

# One-Way ANOVA Test Implementation
class ANOVATest(StatisticalTest):

    def run_test(self, data, *cols, alpha=0.05):
        """Performs one-way ANOVA test for multiple groups."""
        groups = [data[col].dropna() for col in cols]
        f_stat, p_val = stats.f_oneway(*groups)
        print(f"ANOVA F-statistic: {f_stat}, p-value: {p_val}")
        if p_val < alpha:
            print("Reject the null hypothesis: Significant difference between the means of the groups.")
        else:
            print("Fail to reject the null hypothesis: No significant difference between the means of the groups.")
        return f_stat, p_val

# F-Test for equality of variances
class FTest(StatisticalTest):

    def run_test(self, data, col1, col2, alpha=0.05):
        """Performs an F-test for equality of variances between two samples."""
        var1 = np.var(data[col1].dropna(), ddof=1)
        var2 = np.var(data[col2].dropna(), ddof=1)
        f_stat = var1 / var2
        dfn = len(data[col1]) - 1
        dfd = len(data[col2]) - 1
        p_val = 1 - stats.f.cdf(f_stat, dfn, dfd)
        print(f"F-test Statistic: {f_stat}, p-value: {p_val}")
        if p_val < alpha:
            print("Reject the null hypothesis: The variances of the two groups are significantly different.")
        else:
            print("Fail to reject the null hypothesis: The variances of the two groups are not significantly different.")
        return f_stat, p_val

# Levene's Test for equality of variances
class LeveneTest(StatisticalTest):

    def run_test(self, data, *cols, alpha=0.05):
        """Performs Levene's test for equality of variances."""
        groups = [data[col].dropna() for col in cols]
        stat, p_val = stats.levene(*groups)
        print(f"Levene's Test Statistic: {stat}, p-value: {p_val}")
        if p_val < alpha:
            print("Reject the null hypothesis: The variances of the groups are significantly different.")
        else:
            print("Fail to reject the null hypothesis: The variances of the groups are not significantly different.")
        return stat, p_val

# Anderson-Darling Test for normality
class AndersonDarlingTest(StatisticalTest):

    def run_test(self, data, col, alpha=0.05):
        """Performs Anderson-Darling test for normality."""
        result = stats.anderson(data[col].dropna())
        print(f"Anderson-Darling Statistic: {result.statistic}, Critical Values: {result.critical_values}")
        # Interpret results against critical values
        significant = result.statistic > result.critical_values[2]  # 5% significance level
        if significant:
            print("Reject the null hypothesis: The data is not normally distributed.")
        else:
            print("Fail to reject the null hypothesis: The data is normally distributed.")
        return result.statistic, result.critical_values

# Chi-Square Goodness of Fit Test
class GoodnessOfFitTest(StatisticalTest):

    def run_test(self, data, col, expected_probs, alpha=0.05):
        """Performs Chi-Square Goodness of Fit test."""
        observed_counts = data[col].value_counts()
        observed_freq = np.array([observed_counts.get(i, 0) for i in range(len(expected_probs))])
        expected_freq = np.array(expected_probs) * np.sum(observed_freq)
        
        chi2, p = stats.chisquare(f_obs=observed_freq, f_exp=expected_freq)
        print(f"Chi-Square Goodness of Fit Statistic: {chi2}, p-value: {p}")
        if p < alpha:
            print("Reject the null hypothesis: The observed frequencies significantly differ from expected frequencies.")
        else:
            print("Fail to reject the null hypothesis: The observed frequencies do not significantly differ from expected frequencies.")
        return chi2, p

# Strategy Context for running tests
class StatisticalTestContext:
    def __init__(self, strategy: StatisticalTest):
        self._strategy = strategy

    def set_strategy(self, strategy: StatisticalTest):
        self._strategy = strategy

    def execute_test(self, data, *args, alpha=0.05):
        return self._strategy.run_test(data, *args, alpha=alpha)
if __name__ == "__main__":
   
  '''  url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    titanic_data = pd.read_csv(url) 
   
    # Create a statistical test context
    test_context = StatisticalTestContext(ChiSquareTest())

    # Perform Chi-Square Test between 'Survived' and 'Sex'
    print("Chi-Square Test between 'Survived' and 'Sex':")
    chi2_stat, chi2_p = test_context.execute_test(titanic_data, 'Survived', 'Sex')
    
    # Change strategy to Mann-Whitney U Test
    test_context.set_strategy(MannWhitneyUTest())
    
    # Perform Mann-Whitney U Test between 'Age' and 'Survived'
    print("\nMann-Whitney U Test between 'Age' and 'Survived':")
    mw_stat, mw_p = test_context.execute_test(titanic_data, 'Age', 'Survived')
    
    # Change strategy to ANOVA Test
    test_context.set_strategy(ANOVATest())
    
    # Perform ANOVA Test between 'Fare' and 'Pclass'
    print("\nANOVA Test between 'Fare' and 'Pclass':")
    anova_stat, anova_p = test_context.execute_test(titanic_data, 'Fare', 'Pclass')
    '''
    pass
