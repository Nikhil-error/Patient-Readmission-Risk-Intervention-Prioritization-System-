"""
EDA & Visualization Module
Healthcare Analytics: Predicting Hospital Readmissions for Diabetes Patients
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
import os

PALETTE = ['#2196F3', '#F44336', '#4CAF50', '#FF9800', '#9C27B0']
sns.set_theme(style='whitegrid', palette=PALETTE)

os.makedirs('reports/figures', exist_ok=True)


def plot_target_distribution(df, target='readmitted', save=True):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Target Variable Distribution', fontsize=15, fontweight='bold')

    vc = df[target].value_counts()
    axes[0].bar(vc.index, vc.values, color=PALETTE[:len(vc)])
    axes[0].set_title('Readmission Counts')
    axes[0].set_xlabel('Readmission Status')
    axes[0].set_ylabel('Count')
    for i, v in enumerate(vc.values):
        axes[0].text(i, v + 20, str(v), ha='center', fontweight='bold')

    axes[1].pie(vc.values, labels=vc.index, autopct='%1.1f%%',
                colors=PALETTE[:len(vc)], startangle=90)
    axes[1].set_title('Readmission Proportion')

    plt.tight_layout()
    if save:
        plt.savefig('reports/figures/01_target_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 01_target_distribution.png")


def plot_age_distribution(df, save=True):
    fig, ax = plt.subplots(figsize=(12, 5))
    age_order = ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)',
                 '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)']
    age_order = [a for a in age_order if a in df['age'].values]
    
    if age_order:
        sns.countplot(data=df, x='age', hue='readmitted',
                      order=age_order, ax=ax, palette=PALETTE)
        ax.set_title('Readmission by Age Group', fontsize=14, fontweight='bold')
        ax.set_xlabel('Age Group')
        ax.set_ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        if save:
            plt.savefig('reports/figures/02_age_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 02_age_distribution.png")


def plot_numeric_distributions(df, save=True):
    num_cols = ['time_in_hospital', 'num_lab_procedures', 'num_medications',
                'number_inpatient', 'number_diagnoses']
    num_cols = [c for c in num_cols if c in df.columns]

    fig, axes = plt.subplots(1, len(num_cols), figsize=(18, 5))
    fig.suptitle('Numeric Feature Distributions', fontsize=14, fontweight='bold')

    for ax, col in zip(axes, num_cols):
        df.boxplot(column=col, by='readmitted', ax=ax)
        ax.set_title(col.replace('_', ' ').title())
        ax.set_xlabel('Readmission')

    plt.tight_layout()
    if save:
        plt.savefig('reports/figures/03_numeric_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 03_numeric_distributions.png")


def plot_correlation_heatmap(df, save=True):
    num_df = df.select_dtypes(include='number')
    if num_df.shape[1] < 2:
        return
    corr = num_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=0.5, ax=ax)
    ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save:
        plt.savefig('reports/figures/04_correlation_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 04_correlation_heatmap.png")


def plot_roc_curves(results, y_test, save=True):
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = PALETTE

    for (name, res), color in zip(results.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f'{name} (AUC = {roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Classifier')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves — Model Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save:
        plt.savefig('reports/figures/05_roc_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 05_roc_curves.png")


def plot_confusion_matrices(results, y_test, save=True):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    fig.suptitle('Confusion Matrices', fontsize=14, fontweight='bold')

    for ax, (name, res) in zip(axes, results.items()):
        cm = res['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Not Readmitted', 'Readmitted'],
                    yticklabels=['Not Readmitted', 'Readmitted'])
        ax.set_title(name, fontsize=12)
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')

    plt.tight_layout()
    if save:
        plt.savefig('reports/figures/06_confusion_matrices.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 06_confusion_matrices.png")


def plot_feature_importance(importances, top_n=15, save=True):
    n = len(importances)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(8 * n, 6))
    if n == 1:
        axes = [axes]
    fig.suptitle(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')

    for ax, (name, imp) in zip(axes, importances.items()):
        top = imp.head(top_n)
        ax.barh(top['feature'][::-1], top['importance'][::-1], color=PALETTE[0])
        ax.set_title(name, fontsize=12)
        ax.set_xlabel('Importance')

    plt.tight_layout()
    if save:
        plt.savefig('reports/figures/07_feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 07_feature_importance.png")


def plot_model_comparison(summary_df, save=True):
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(metrics))
    width = 0.25
    n_models = len(summary_df)

    for i, (_, row) in enumerate(summary_df.iterrows()):
        vals = [row[m] for m in metrics]
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=row['Model'],
                      color=PALETTE[i % len(PALETTE)], alpha=0.85)

    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    if save:
        plt.savefig('reports/figures/08_model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 08_model_comparison.png")
