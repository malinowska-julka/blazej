import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


# from chatGPT
def cohens_d(x, groups):
    """
    Compute Cohen's d for two groups.

    Parameters:
        x (pd.Series): Numerical variable.
        groups (pd.Series): Categorical variable with exactly two unique groups.

    Returns:
        float: Cohen's d value.
    """
    # Ensure we have exactly two groups
    levels = groups.unique()
    if len(levels) != 2:
        raise ValueError("The grouping variable must have exactly two unique values.")

    # Split data by group
    x1 = x[groups == levels[0]]
    x2 = x[groups == levels[1]]

    # Compute means and SDs
    mean1, mean2 = np.mean(x1), np.mean(x2)
    sd1, sd2 = np.std(x1, ddof=1), np.std(x2, ddof=1)
    n1, n2 = len(x1), len(x2)

    # Compute pooled SD
    sd_pooled = np.sqrt(((n1 - 1) * sd1**2 + (n2 - 1) * sd2**2) / (n1 + n2 - 2))

    # Compute Cohen's d
    d = (mean1 - mean2) / sd_pooled
    return d


def cramers_v(x, y):
    """
    Compute Cramér's V statistic for categorical-categorical association.

    Parameters:
        x (pd.Series): Categorical variable.
        y (pd.Series): Categorical variable.

    Returns:
        float: Cramér's V value between 0 and 1.
    """
    # Create a contingency table
    cont_table = pd.crosstab(x, y)

    # Compute chi-squared statistic
    chi2 = chi2_contingency(cont_table)[0]

    # Get sample size and minimum dimension
    n = cont_table.sum().sum()
    k = min(cont_table.shape) - 1

    # Compute Cramér's V
    v = np.sqrt(chi2 / (n * k))
    return v
