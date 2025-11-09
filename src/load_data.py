import pandas as pd

def load_telco_data(path: str) -> pd.DataFrame:
    """Load Telco Customer Churn dataset."""
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["Churn"] = df["Churn"].apply(lambda x: 1 if x == "Yes" else 0)
    return df
