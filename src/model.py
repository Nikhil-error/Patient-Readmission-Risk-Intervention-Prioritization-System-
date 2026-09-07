"""
Model Training Module
Healthcare Analytics: Predicting Hospital Readmissions for Diabetes Patients
"""

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')


MODELS = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, class_weight='balanced', random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, class_weight='balanced',
        max_depth=8, random_state=42, n_jobs=-1
    ),
    "XGBoost": XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        scale_pos_weight=3, use_label_encoder=False,
        eval_metric='logloss', random_state=42
    )
}


def train_and_evaluate(X_train, X_test, y_train, y_test, feature_names=None):
    """Train all models and return results."""
    results = {}

    for name, model in MODELS.items():
        print(f"\n{'='*50}")
        print(f"Training: {name}")
        print('='*50)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = {
            'model': model,
            'y_pred': y_pred,
            'y_prob': y_prob,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_prob),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }

        results[name] = metrics

        print(f"Accuracy : {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall   : {metrics['recall']:.4f}")
        print(f"F1-Score : {metrics['f1']:.4f}")
        print(f"ROC-AUC  : {metrics['roc_auc']:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred,
              target_names=['Not Readmitted', 'Readmitted <30d']))

    return results


def get_feature_importance(results, feature_names):
    """Extract feature importances from tree-based models."""
    importances = {}
    for name, res in results.items():
        model = res['model']
        if hasattr(model, 'feature_importances_'):
            imp = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            importances[name] = imp
        elif hasattr(model, 'coef_'):
            imp = pd.DataFrame({
                'feature': feature_names,
                'importance': np.abs(model.coef_[0])
            }).sort_values('importance', ascending=False)
            importances[name] = imp
    return importances


def save_best_model(results, output_dir='models'):
    """Save the best model by ROC-AUC score."""
    os.makedirs(output_dir, exist_ok=True)
    best_name = max(results, key=lambda k: results[k]['roc_auc'])
    best_model = results[best_name]['model']
    path = os.path.join(output_dir, 'best_model.pkl')
    with open(path, 'wb') as f:
        pickle.dump(best_model, f)
    print(f"\nBest model: {best_name} (AUC={results[best_name]['roc_auc']:.4f})")
    print(f"Saved to: {path}")
    return best_name, path


def summary_table(results):
    """Return a DataFrame summarising all model metrics."""
    rows = []
    for name, res in results.items():
        rows.append({
            'Model': name,
            'Accuracy': round(res['accuracy'], 4),
            'Precision': round(res['precision'], 4),
            'Recall': round(res['recall'], 4),
            'F1-Score': round(res['f1'], 4),
            'ROC-AUC': round(res['roc_auc'], 4),
        })
    return pd.DataFrame(rows).sort_values('ROC-AUC', ascending=False)
