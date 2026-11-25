# Stroke Data Analysis

A Python course project exploring a public stroke dataset with exploratory data analysis, clustering, regression and binary classification.

## Analysis

| Stage | Approach |
| --- | --- |
| Data preparation | Inspect types, clean categorical values, drop duplicate and incomplete records |
| Exploration | Compare observed stroke rates across groups using pivot tables and charts |
| Clustering | Explore groups using K-means and scaled features |
| Regression | Model average glucose level using age and BMI with ordinary least squares |
| Classification | Compare decision trees, random forest, XGBoost and a small neural network |

The tree-based models report accuracy, precision, recall, F1, ROC AUC and confusion matrices on a stratified test split. Stroke cases are less frequent, so recall and the confusion matrix give context to accuracy. This is an educational analysis. The neural network also uses the test split for epoch validation, so its results are exploratory.

## Run locally

1. Download [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) and put `healthcare-dataset-stroke-data.csv` next to `main.py`.
2. With Python 3.12 installed, run these commands from the project folder in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

The script prints summaries and model metrics. Close each chart window to continue to the next stage.
