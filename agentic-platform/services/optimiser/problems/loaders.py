"""
Dataset loaders for ML problems.

Provides functions to load and preprocess datasets for hyperparameter optimization.
"""

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Data directory relative to project root
DATA_DIR = Path(__file__).parent.parent.parent / "data"

TITANIC_URL = (
    "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
)


def ensure_data_dir() -> None:
    """Create data directory if it doesn't exist."""
    DATA_DIR.mkdir(exist_ok=True)


def download_titanic() -> Path:
    """
    Download titanic.csv if not already present.

    Returns:
        Path to the downloaded file
    """
    ensure_data_dir()
    filepath = DATA_DIR / "titanic.csv"
    if not filepath.exists():
        print(f"Downloading Titanic dataset to {filepath}...")
        urllib.request.urlretrieve(TITANIC_URL, filepath)
        print("Download complete.")
    return filepath


def load_titanic(
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and preprocess Titanic dataset.

    Features used: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
    Target: Survived (0/1)

    Preprocessing:
    - Fill missing Age with median
    - Fill missing Embarked with mode ('S')
    - Encode Sex (male=1, female=0)
    - Encode Embarked (S=0, C=1, Q=2)
    - Scale numeric features (Age, Fare)

    Args:
        test_size: Fraction of data for validation set
        random_state: Random seed for reproducibility

    Returns:
        (X_train, X_val, y_train, y_val) as numpy arrays
    """
    filepath = download_titanic()
    df = pd.read_csv(filepath)

    # Select features
    features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    target = "Survived"

    df = df[features + [target]].copy()

    # Handle missing values
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna("S")

    # Encode categorical features
    df["Sex"] = LabelEncoder().fit_transform(df["Sex"])  # male=1, female=0
    df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

    # Split features and target
    X = df[features].values.astype(np.float64)
    y = df[target].values

    # Scale numeric features (Age at index 2, Fare at index 5)
    scaler = StandardScaler()
    X[:, [2, 5]] = scaler.fit_transform(X[:, [2, 5]])

    # Train/validation split with stratification
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def load_fraud(
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and preprocess Credit Card Fraud dataset (sampled version).

    Features: V1-V28 (PCA transformed), Amount
    Target: Class (0=legitimate, 1=fraud)

    Preprocessing:
    - Drop Time column (not useful for prediction)
    - Scale Amount column
    - Stratified split (preserve class imbalance)

    Note: Original dataset is ~0.17% fraud, our sampled version is ~10% fraud.

    Args:
        test_size: Fraction of data for validation set
        random_state: Random seed for reproducibility

    Returns:
        (X_train, X_val, y_train, y_val) as numpy arrays
    """
    filepath = DATA_DIR / "creditcard_sampled.csv"
    if not filepath.exists():
        raise FileNotFoundError(
            f"Credit card fraud dataset not found at {filepath}. "
            "Run 'python scripts/sample_fraud_data.py' to create it."
        )

    df = pd.read_csv(filepath)

    # Drop Time column, separate features and target
    X = df.drop(["Time", "Class"], axis=1).values.astype(np.float64)
    y = df["Class"].values

    # Scale Amount column (last column, index 28)
    scaler = StandardScaler()
    X[:, -1] = scaler.fit_transform(X[:, -1:]).flatten()

    # Train/validation split with stratification
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def load_housing(
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and preprocess California Housing dataset.

    Features: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude
    Target: Median house value (in $100,000s)

    Preprocessing:
    - StandardScaler on all features
    - No missing values to handle

    Args:
        test_size: Fraction of data for validation set
        random_state: Random seed for reproducibility

    Returns:
        (X_train, X_val, y_train, y_val) as numpy arrays
    """
    from sklearn.datasets import fetch_california_housing

    data = fetch_california_housing()
    X = data.data.astype(np.float64)
    y = data.target

    # Scale all features
    X = StandardScaler().fit_transform(X)

    # Train/validation split (no stratification for regression)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
