# Predictive Modeling for term deposit subscriptions

This project addresses the challenge faced by the banking industry in identifying customers likely to subscribe to term deposits. Using machine learning, it helps banks optimize marketing efforts and reduce costs.

## Overview

The dataset includes observations from direct phone marketing campaigns conducted by a banking institution. It features 40,841 samples with 16 customer attributes and a binary target variable indicating whether the customer subscribed to a term deposit.

The solution involves building predictive models to identify potential subscribers using:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

The Random Forest model was selected for deployment due to its superior accuracy and feature importance insights.

## Features

- **Customer demographics** (e.g., age, marital status, job type)
- **Financial information** (e.g., account balance, loan status)
- **Campaign details** (e.g., contact duration, number of contacts)
- **Outcome of the previous campaign**

## Key Steps

### Data Preprocessing

- Removed uninformative values (`unknown`, `others`) and outliers.
- Split data into training and validation sets.

### Modeling

- Implemented Logistic Regression, Decision Tree, and Random Forest models.
- Evaluated models using accuracy, AUC-ROC, and classification reports.

### Visualization

- Age distribution
- Correlation heatmap
- Feature importance

### Deployment

- Deployed the Random Forest model with Gradio for real-time predictions.

## Dependencies

- **Python** 3.x
- Required libraries: `numpy`, `pandas`, `matplotlib`, `seaborn`, `sklearn`, `gradio`, `pickle`, `graphviz`

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/term-deposit-prediction.git
   cd term-deposit-prediction


