
# Customer Statistical Analysis Report

**Project:** AI Data Intelligence Platform  
**Analysis:** Customer Purchase Behavior  
**Dataset Type:** Synthetic demonstration dataset  
**Status:** Statistical analysis completed

---

## 1. Executive Summary

This report analyzes customer purchase behavior using descriptive
statistics, independent group comparisons, hypothesis testing,
and effect size measurements.

The dataset is synthetic and does not represent real customers
or commercial transactions.

## 2. Dataset Overview

| Metric | Value |
|---|---:|
| Total cleaned records | 998 |
| Buyers | 396 |
| Non-buyers | 602 |
| Purchase rate | 39.68% |
| Features analyzed | 6 |

## 3. Statistical Methodology

The following methods were applied:

- Welch's independent samples t-test
- Mann-Whitney U test
- Bonferroni multiple-testing correction
- Cohen's d effect size
- Rank-biserial correlation

A significance threshold of 0.05 was used after
Bonferroni correction.

## 4. Customer Group Comparison

Buyers and non-buyers were compared across six numerical features.

| Feature | Buyer Mean | Non-buyer Mean | Mean Difference |
|---|---:|---:|---:|
| age | 42.460 | 45.141 | -2.682 |
| annual_income | 56823.178 | 53461.808 | 3361.370 |
| website_visits | 5.061 | 4.822 | 0.238 |
| previous_purchases | 2.995 | 2.246 | 0.749 |
| avg_session_minutes | 7.969 | 7.253 | 0.716 |
| marketing_emails_opened | 3.672 | 3.402 | 0.270 |


## 5. Hypothesis Testing Results

Adjusted p-values account for six feature comparisons.

| Feature | Welch Adjusted p | Welch Significant | Mann-Whitney Adjusted p | Mann-Whitney Significant |
|---|---:|---|---:|---|
| age | 0.037000 | Yes | 0.034999 | Yes |
| annual_income | 0.266778 | No | 0.171705 | No |
| website_visits | 0.650590 | No | 0.642047 | No |
| previous_purchases | 0.000000 | Yes | 0.000000 | Yes |
| avg_session_minutes | 0.199284 | No | 0.191044 | No |
| marketing_emails_opened | 0.036499 | Yes | 0.020455 | Yes |


## 6. Effect Size Analysis

Cohen's d measures standardized mean differences.
Rank-biserial correlation measures rank-based group differences.

| Feature | Cohen's d | Rank-Biserial Correlation |
|---|---:|---:|
| age | -0.17728 | -0.10302 |
| annual_income | 0.13147 | 0.08181 |
| website_visits | 0.10457 | 0.05967 |
| previous_purchases | 0.47080 | 0.24368 |
| avg_session_minutes | 0.14157 | 0.08023 |
| marketing_emails_opened | 0.17804 | 0.10719 |


## 7. Key Findings

- The feature with the largest absolute Cohen's d is
  **previous_purchases**
  (d = 0.4708).

- The feature with the largest absolute rank-biserial correlation is
  **previous_purchases**
  (r = 0.2437).

- Previous purchases showed the strongest positive effect
  in the earlier group comparison analysis.

- Statistical significance and practical business importance
  must be evaluated separately.

## 8. Business Interpretation

Previous purchase history appears to be a useful candidate signal
for customer purchase prediction in this synthetic dataset.

This finding may motivate further investigation of customer
purchase history, repeat-purchase behavior, and customer
segmentation.

These observations do not establish causation.

## 9. Limitations

1. The dataset is synthetic.
2. Statistical associations do not establish causal relationships.
3. Multiple-testing correction was applied to the six feature tests.
4. Cohen's d uses a pooled standard deviation and may be less
   suitable when group variances differ substantially.
5. Statistical results alone do not establish predictive performance.
6. Model validation on independent, representative data is required
   before making real business decisions.

## 10. Recommended Next Steps

1. Complete feature engineering.
2. Define a machine learning prediction target.
3. Establish a train-validation-test strategy.
4. Build a baseline classification model.
5. Evaluate generalization using suitable classification metrics.

---

**Report Status:** Completed  
**Project:** AI Data Intelligence Platform  
**Data Disclaimer:** Synthetic data used for engineering demonstration.
