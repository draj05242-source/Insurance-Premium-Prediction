# Insurance-Premium-Prediction
> Binary Classification · Logistic Regression · Insurance Analytics · 2025
Predicting which customers will pay their insurance premium using Logistic Regression — with statistical significance testing, optimal threshold selection via Youden's J, and actionable business recommendations for revenue risk management.

## Dataset
Customer-level insurance records · Features include Age Band, Gender, Marital Status,
Vehicle Age, and policy details · Binary target: `Premium paid` (Yes / No)

## My Approach
- Dropped non-predictive `Customer_ID` column at the start of the pipeline
- Removed outliers in `Vehicle Age` using the IQR method (Q1 − 1.5×IQR / Q3 + 1.5×IQR)
- One-hot encoded all categorical variables with `drop_first=True` to avoid multicollinearity
- Fitted `statsmodels.Logit` for full statistical output — coefficients, p-values, z-scores, confidence intervals
- Pruned `Age Band_B` after identifying it as statistically insignificant (p > 0.05) from the summary table
- Evaluated at two thresholds: default 0.5 and the Youden's J optimal cut-off (maximises Sensitivity + Specificity − 1)


**Why Youden's J?** The default 0.5 cut-off is rarely optimal on imbalanced insurance data.
Youden's J finds the threshold that simultaneously maximises true positive and true negative
rates — a standard technique in actuarial and medical classification.

## Business Recommendations
- Use the Youden-optimal threshold as the operational cut-off for flagging non-payers
- If revenue recovery is the priority (missing a non-payer is costly), lower the threshold to increase Sensitivity
- If customer relationship damage is the concern (false alarms cause friction), raise the threshold to protect Specificity
- Statistically significant features from the logit summary directly indicate which customer segments carry the highest non-payment risk — target those for premium reminder campaigns

## Tech Stack
`Python` `Pandas` `NumPy` `Statsmodels` `Scikit-learn` `Matplotlib` `Seaborn`

## What I'd Do Differently
- Apply backward stepwise elimination in a loop instead of manually dropping `Age Band_B` — iteratively remove the highest p-value feature until all remaining predictors are significant (p < 0.05)
- Extend IQR outlier removal to *all* numerical features, not just `Vehicle Age` — unchecked outliers can distort logistic regression coefficients
- Add a calibration plot (reliability diagram) to verify that predicted probabilities are trustworthy, not just discrimination metrics like AUC
- Use 5-fold stratified cross-validation instead of a single 80/20 split for more stable metric estimates
- Include a dummy classifier baseline — if 80% of customers already pay, a model that always predicts "Yes" also scores 80% accuracy; baseline is essential context
- Test Random Forest or XGBoost alongside logistic regression to quantify the interpretability–performance trade-off

## Run It Yourself
```bash
pip install pandas numpy statsmodels scikit-learn matplotlib seaborn
```
Place `Insurance data.csv` in the working directory, then open `revised_insurance_logit.ipynb` in Jupyter or Google Colab and run all cells.
