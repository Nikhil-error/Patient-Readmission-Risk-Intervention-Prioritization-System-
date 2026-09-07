"""
Main Pipeline
Healthcare Analytics: Predicting Hospital Readmissions for Diabetes Patients
Dataset: UCI ML Repository — Diabetes 130-US Hospitals (1999-2008)
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
from src.preprocess import prepare_data, load_data, clean_data, encode_target
from src.model import train_and_evaluate, get_feature_importance, save_best_model, summary_table
from src.visualize import (
    plot_target_distribution, plot_age_distribution, plot_numeric_distributions,
    plot_correlation_heatmap, plot_roc_curves, plot_confusion_matrices,
    plot_feature_importance, plot_model_comparison
)


def main():
    print("=" * 60)
    print("  Healthcare Analytics: Diabetes Readmission Prediction")
    print("=" * 60)

    DATA_PATH = 'data/diabetic_data.csv'

    # ── EDA ──────────────────────────────────────────────────────
    print("\n[1/4] Running Exploratory Data Analysis...")
    df_raw = load_data(DATA_PATH)
    df_clean = clean_data(df_raw.copy())
    df_eda = encode_target(df_clean.copy(), target_col='readmitted')

    # Re-map readmitted back to string for EDA plots
    df_eda_str = df_clean.copy()
    plot_target_distribution(df_eda_str)
    plot_age_distribution(df_eda_str)
    plot_numeric_distributions(df_eda_str)

    # Encode for numeric correlation heatmap
    from src.preprocess import encode_features
    df_enc = encode_features(df_eda.copy())
    plot_correlation_heatmap(df_enc)

    # ── Prepare data ─────────────────────────────────────────────
    print("\n[2/4] Preparing data for modelling...")
    X_train, X_test, y_train, y_test, feature_names, scaler = prepare_data(DATA_PATH)

    # ── Train & Evaluate ─────────────────────────────────────────
    print("\n[3/4] Training models...")
    results = train_and_evaluate(X_train, X_test, y_train, y_test, feature_names)

    # ── Visualise results ────────────────────────────────────────
    print("\n[4/4] Generating result plots...")
    plot_roc_curves(results, y_test)
    plot_confusion_matrices(results, y_test)

    importances = get_feature_importance(results, feature_names)
    plot_feature_importance(importances)

    summary = summary_table(results)
    plot_model_comparison(summary)

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  MODEL SUMMARY")
    print("=" * 60)
    print(summary.to_string(index=False))

    best_name, model_path = save_best_model(results)

    print("\nAll figures saved to: reports/figures/")
    print("Best model saved to:", model_path)
    print("\nProject complete!")


if __name__ == '__main__':
    main()
