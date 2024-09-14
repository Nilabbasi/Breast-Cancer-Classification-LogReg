# Logistic Regression Classifier

This project implements a Logistic Regression model for binary classification using both a custom implementation with NumPy and the scikit-learn library. The dataset used is the Breast Cancer Wisconsin (Diagnostic) dataset, which contains various features related to breast cancer tumors.

## Table of Contents

- [Introduction](#introduction)
- [Libraries Used](#libraries-used)
- [Dataset](#dataset)
- [Implementation](#implementation)
  - [Custom Logistic Regression Model](#custom-logistic-regression-model)
  - [Scikit-learn Logistic Regression Model](#scikit-learn-logistic-regression-model)
- [Results](#results)
- [ROC Curve and Evaluation Metrics](#roc-curve-and-evaluation-metrics)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Introduction

The goal of this project is to develop a logistic regression classifier for predicting whether a tumor is malignant or benign based on its features. The project includes data visualization, model training, evaluation, and comparison of two logistic regression implementations.

## Libraries Used

This project uses the following libraries:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `sklearn`

```python
import math
import statistics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
```

## Dataset

The dataset used in this project is the Breast Cancer Wisconsin (Diagnostic) dataset from the `sklearn` library. It contains 569 samples with 30 features, indicating various characteristics of tumors, and a binary target variable indicating whether a tumor is malignant (1) or benign (0).

## Implementation

### Custom Logistic Regression Model

The first part of the project includes a custom implementation of a logistic regression model using NumPy. The model is trained using gradient descent and evaluated on a test dataset. 

Key components include:
- Sigmoid function for prediction
- Gradient descent optimization for weight and bias updates

### Scikit-learn Logistic Regression Model

The second part uses the `LogisticRegression` class from the `sklearn` library. The model is trained on the same dataset, and various performance metrics are calculated to compare with the custom implementation.

## Results

The performance of both implementations is evaluated based on:
- Accuracy
- True Positives (TP)
- False Positives (FP)
- True Negatives (TN)
- False Negatives (FN)

### Example Results

- **Custom Implementation:**
  - True Positives (TP): 87
  - False Positives (FP): 1
  - True Negatives (TN): 25
  - False Negatives (FN): 1

- **Scikit-learn Implementation:**
  - True Positives (TP): 88
  - False Positives (FP): 0
  - True Negatives (TN): 26
  - False Negatives (FN): 0

## ROC Curve and Evaluation Metrics

The project also includes an analysis of the ROC curve to evaluate the model's performance at different thresholds. Metrics such as sensitivity, specificity, F1 Score, and precision are reported.

## Installation

To run this project, you need to have Python installed along with the required libraries. You can install the necessary packages using pip:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Usage

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd logistic-regression-classifier
   ```

2. Run the Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

3. Follow the cells in the notebook to execute the Logistic Regression implementations and visualize the results.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

