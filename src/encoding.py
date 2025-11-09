import pandas as pd
from sklearn.preprocessing import LabelEncoder


def label_encode_binary(df):
    """Encode binary categorical variables."""
    binary_cols = [col for col in df.columns if df[col].dtype == "O" and df[col].nunique() == 2]
    for col in binary_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    return df


def one_hot_encode(df, exclude_cols=None):
    """Perform one-hot encoding on categorical features."""
    exclude_cols = exclude_cols or []
    cat_cols = [col for col in df.columns if df[col].dtype == "O" and col not in exclude_cols]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df
