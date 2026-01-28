# Distribution of a single variable

## Measures of central tendency and positional measures
* arithmetic mean
* median
* geometric mean (useful for calculation of average rate of growth):

    $$\left(\prod_{i=1}^n{x_i}\right)^{1/n}$$
    
    or   

    $$\exp\left(\frac{1}{n}\left(\sum_{i=1}^n{\ln(x_i)}\right)\right)$$

* trimmed mean
* positional measures - quantiles: quartiles, deciles, percentiles

## Measures of variability / dispersion

* variance, standard deviation
* IQR (interquartile range)
* mean absolute difference
* coefficient of variation: standard deviation divided by the mean

## Measures of shape of the distribution

* skewness (measure of asymmetry)
* kurtosis (measure of tailedness, extremity of tails)

# Association measures between quantitative and binary variable (say, height and gender, hand size and gender...)

* Cohen's d (difference between the means adjusted for the pooled standard deviation)
* point-biserial correlation coefficient (Pearson correlation when one of the variables is 0/1)
* probability of superiority (AUC, C-statistic, common language effect size...)
* Somers D = 2*AUC-1

## Cohen's d

$$ \text{Cohen's d} = \frac{\bar{x}_1 - \bar{x}_2}{\text{sd}_{pooled}}$$
$$\text{sd}_{pooled} = \sqrt{\frac{(n_1-1)\cdot s_1^2+(n_2-1)\cdot s_2^2}{n_1+n_2-2}}$$
$$s_1 = \sqrt{\frac{\sum_{i=1}^{n_1} (x_i-\bar{x})^2}{n_1-1}}$$

## Cramers-V correlation between two categorical

```python
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

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

print(cramers_v(d['eye2'], d['gender']))
```
## AUC 

We have two variables: X and Y. X is numeric or ordinal. Y is binary. (Y=0 or Y=1)

AUC(X, Y)

AUC - technically: area under the ROC curve. 

Equivalent - via probability. Let's sample one observation ($X_0$, Y), where Y=0 and one observation ($X_1$, Y) where Y=1. 

$$AUC = P(X_1 > X_0) + \frac{1}{2}P(X_0 = X_1)$$

When there is no association (~correlation) between X and Y, AUC = 0.5. 

Somers' D (Gini in banking) = $2 AUC - 1 $

$$D = P(X_1 > X_0) - P(X_1 < X_0)$$