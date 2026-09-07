"""
Data Preprocessing Module
Healthcare Analytics: Predicting Hospital Readmissions for Diabetes Patients
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


def load_data(filepath: str) -> pd.DataFrame:
    """Load the diabetes dataset."""
    df = pd.read_csv(filepath)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess the raw dataset."""
    df = df.copy()

    # Replace '?' with NaN
    df.replace('?', np.nan, inplace=True)

    # Drop high-missing columns (weight, payer_code, medical_specialty if present)
    high_missing = [col for col in ['weight', 'payer_code', 'medical_specialty']
                    if col in df.columns and df[col].isna().mean() > 0.4]
    if high_missing:
        df.drop(columns=high_missing, inplace=True)
        print(f"Dropped high-missing columns: {high_missing}")

    # Remove duplicate patients (keep first encounter)
    if 'patient_nbr' in df.columns:
        df.drop_duplicates(subset='patient_nbr', keep='first', inplace=True)
        print(f"After dedup: {df.shape[0]} rows")

    # Drop ID columns
    id_cols = [c for c in ['encounter_id', 'patient_nbr'] if c in df.columns]
    df.drop(columns=id_cols, inplace=True)

    # Fill remaining missing values
    for col in df.select_dtypes(include='object').columns:
        df[col].fillna(df[col].mode()[0], inplace=True)
    for col in df.select_dtypes(include='number').columns:
        df[col].fillna(df[col].median(), inplace=True)

    return df


def encode_target(df: pd.DataFrame, target_col: str = 'readmitted') -> pd.DataFrame:
    """Encode target: readmitted within 30 days = 1, else = 0."""
    df = df.copy()
    df[target_col] = (df[target_col] == '<30').astype(int)
    print(f"Target distribution:\n{df[target_col].value_counts()}")
    return df


def encode_features(df: pd.DataFrame, target_col: str = 'readmitted'):
    """Encode categorical features and scale numerics."""
    df = df.copy()

    # Age range → midpoint
    age_map = {
        '[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35,
        '[40-50)': 45, '[50-60)': 55, '[60-70)': 65, '[70-80)': 75,
        '[80-90)': 85, '[90-100)': 95
    }
    if 'age' in df.columns:
        df['age'] = df['age'].map(age_map).fillna(df['age'])

    # Label encode remaining categoricals
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    if target_col in cat_cols:
        cat_cols.remove(target_col)

    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    return df


def prepare_data(filepath: str, test_size: float = 0.2, random_state: int = 42):
    """Full pipeline: load → clean → encode → split."""
    df = load_data(filepath)
    df = clean_data(df)
    df = encode_target(df)
    df = encode_features(df)

    X = df.drop(columns=['readmitted'])
    y = df['readmitted']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")
    print(f"Positive rate (train): {y_train.mean():.2%}")

    return X_train_sc, X_test_sc, y_train, y_test, X.columns.tolist(), scaler
