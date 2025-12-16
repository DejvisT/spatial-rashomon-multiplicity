"""
Dataset loading and preprocessing utilities.
"""
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer, fetch_openml
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_selector
from sklearn.model_selection import train_test_split


def load_dataset(name: str, data_dir: str = "data"):
    """
    Load a dataset by name.
    
    Parameters
    ----------
    name : str
        Dataset name. Available: 'compas', 'german_credit', 'adult', 'breast_cancer'
    data_dir : str
        Directory containing local data files (for COMPAS)
        
    Returns
    -------
    X : pd.DataFrame
        Feature matrix (cleaned but NOT imputed/encoded)
    y : pd.Series
        Binary target variable
    """
    if name == "compas":
        df = pd.read_csv(f"{data_dir}/compas-scores-two-years.csv")
        y = df["two_year_recid"]
        X = df[["age", "sex", "race", "priors_count", "c_charge_degree"]]
        
    elif name == "german_credit":
        data = fetch_openml("credit-g", version=1, as_frame=True, parser="pandas")
        X, y = data.data, (data.target == "good").astype(int)
        
    elif name == "adult":
        data = fetch_openml(name="adult", version=2, as_frame=True, parser="pandas")
        X, y = data.data, (data.target == ">50K").astype(int)
        
    elif name == "breast_cancer":
        X, y = load_breast_cancer(return_X_y=True, as_frame=True)
        
    else:
        raise ValueError(
            f"Unknown dataset: {name}. "
            f"Available: compas, german_credit, adult, breast_cancer"
        )
    
    return X, y


def prepare_data(name: str, test_size: float = 0.3, random_state: int = 42, data_dir: str = "data"):
    """
    Load dataset and split into train/test with scaling.
    
    Parameters
    ----------
    name : str
        Dataset name
    test_size : float
        Fraction of data for test set
    random_state : int
        Random seed for reproducibility
    data_dir : str
        Directory containing local data files
        
    Returns
    -------
    dict with keys:
        - X_train, X_test, y_train, y_test: processed splits (imputed + one-hot encoded; numeric)
        - X_train_scaled, X_test_scaled: scaled features
        - feature_names: list of feature names
        - preprocessor: fitted ColumnTransformer (fit on X_train only)
        - scaler: fitted StandardScaler
    """
    X, y = load_dataset(name, data_dir)

    # Train-test split
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size, 
        stratify=y, 
        random_state=random_state
    )

    # Strip whitespace in categorical columns
    X_train_df = X_train_df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)
    X_test_df = X_test_df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)

    # Identify categorical and numeric columns
    cat_cols = make_column_selector(dtype_include=["object", "category"])(X_train_df)
    num_cols = make_column_selector(dtype_exclude=["object", "category"])(X_train_df)

    # Ensure numeric columns are numeric
    if len(num_cols) > 0:
        X_train_df[num_cols] = X_train_df[num_cols].apply(pd.to_numeric, errors="coerce")
        X_test_df[num_cols] = X_test_df[num_cols].apply(pd.to_numeric, errors="coerce")

    # Fit preprocessing
    numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(drop=None, sparse_output=False, handle_unknown="ignore")),
        ]
    )

    transformers = []
    if len(num_cols) > 0:
        transformers.append(("num", numeric_transformer, num_cols))
    if len(cat_cols) > 0:
        transformers.append(("cat", categorical_transformer, cat_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    X_train_arr = preprocessor.fit_transform(X_train_df)
    X_test_arr = preprocessor.transform(X_test_df)

    # Feature names (remove "num__"/"cat__" prefixes for compatibility)
    feature_names_out = list(preprocessor.get_feature_names_out())
    feature_names = [n.split("__", 1)[1] if "__" in n else n for n in feature_names_out]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_arr)
    X_test_scaled = scaler.transform(X_test_arr)
    
    print(f"✅ Loaded {name}: {X.shape[0]} samples")
    print(f"   Features after preprocessing: {len(feature_names)}")
    print(f"   Train: {len(X_train_df)}, Test: {len(X_test_df)}")
    
    return {
        'X_train': X_train_arr,
        'X_test': X_test_arr,
        'y_train': y_train,
        'y_test': y_test,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'feature_names': feature_names,
        'preprocessor': preprocessor,
        'scaler': scaler
    }

