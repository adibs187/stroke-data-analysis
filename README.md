# Stroke Data Analysis

## Overview

A Python analysis of a public stroke dataset. The project examines how recorded stroke rates vary across patient groups and compares classification models on a stratified train/test split.

## Features

- Clean the dataset and visualize differences across demographic and health groups.
- Explore groups with K-means and model average glucose level using linear regression.
- Compare decision trees, random forest, and XGBoost using precision, recall, F1, ROC AUC, and confusion matrices.
- Explore a small neural network. Its validation uses the test split, so its results are exploratory.

## Tech Stack

Python 3.12, pandas, Matplotlib, Seaborn, scikit-learn, statsmodels, XGBoost, and TensorFlow/Keras.

## Getting Started

1. Download the CSV from the [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset).
2. Place `healthcare-dataset-stroke-data.csv` next to `analyze_stroke_data.py`.
3. From the repository root, run the following in PowerShell.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe analyze_stroke_data.py
```

The script prints summaries and evaluation metrics in the terminal. Close each chart window to continue.

## Repository Structure

| Path | Purpose |
| --- | --- |
| `analyze_stroke_data.py` | Data cleaning, charts, and model evaluation |
| `requirements.txt` | Python dependencies |
