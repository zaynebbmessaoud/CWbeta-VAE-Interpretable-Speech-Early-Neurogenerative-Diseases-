# -*- coding: utf-8 -*-
"""
Created on Mon May 25 13:12:32 2026

@author: user
"""

#-------------
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, recall_score, 
    matthews_corrcoef, make_scorer
)
import matplotlib.pyplot as plt
import seaborn as sns


# 1. Mise à l'échelle (Standardization)
scaler = StandardScaler()

mu_scaled = scaler.fit_transform(mu_vectors)

print("Données prêtes pour l'Ensemble Classifier (XGBoost, SVM, Stacking).")

# 1. Setup Metrics and Scorer
# MCC is excellent for evaluating the quality of the classification
mcc_scorer = make_scorer(matthews_corrcoef)

# 2. Define the XGBoost Grid
param_grid = {
    'n_estimators': [100, 150],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 0.9],
    'colsample_bytree': [0.8, 0.9]
}

# 3. Initialize Stratified 5-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 4. Cross-Validation Loop to collect all metrics
fold_metrics = {'accuracy': [], 'f1': [], 'mcc': [], 'recall': []}
all_y_true = []
all_y_pred = []

for train_index, test_index in skf.split(mu_scaled, labels):
    #X_train, X_test = mu_scaled[train_index], mu_scaled[test_index]
    #y_train, y_test = labels[train_index], labels[test_index]


  
    # Fine-tuning inside the fold
    clf = GridSearchCV(
        xgb.XGBClassifier(objective='multi:softprob', num_class=3, random_state=42, eval_metric='mlogloss'),
        param_grid, cv=3, scoring='f1_weighted', n_jobs=-1
    )
    clf.fit(X_train, y_train)
    
    # Predictions
    y_pred = clf.predict(X_test)
    
    # Store for global confusion matrix
    all_y_true.extend(y_test)
    all_y_pred.extend(y_pred)
    
    # Calculate Fold Metrics including Accuracy
    fold_metrics['accuracy'].append(accuracy_score(y_test, y_pred))
    fold_metrics['f1'].append(f1_score(y_test, y_pred, average='weighted'))
    fold_metrics['mcc'].append(matthews_corrcoef(y_test, y_pred))
    fold_metrics['recall'].append(recall_score(y_test, y_pred, average='weighted'))

# 5. Summary Statistics
# Added print statement for Accuracy
print(f"Mean Accuracy: {np.mean(fold_metrics['accuracy']):.2f} (+/- {np.std(fold_metrics['accuracy']):.2f})")
print(f"Mean F1-Score: {np.mean(fold_metrics['f1']):.2f} (+/- {np.std(fold_metrics['f1']):.2f})")
print(f"Mean MCC: {np.mean(fold_metrics['mcc']):.2f} (+/- {np.std(fold_metrics['mcc']):.2f})")
print(f"Mean Recall: {np.mean(fold_metrics['recall']):.2f} (+/- {np.std(fold_metrics['recall']):.2f})")

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['HC', 'PD', 'TE'], 
                yticklabels=['HC', 'PD', 'TE'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Consolidated Confusion Matrix (5-Fold CV)')
    plt.show()

plot_confusion_matrix(all_y_true, all_y_pred)

