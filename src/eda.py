import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def check_df(df: pd.DataFrame, head=5):
    """Quick look at dataframe structure."""
    print("Shape:", df.shape)
    print("Types:\n", df.dtypes)
    print("\nHead:\n", df.head(head))
    print("\nMissing Values:\n", df.isnull().sum())
    print("\nQuantiles:\n", df.describe(percentiles=[0.05, 0.5, 0.95]).T)


def grab_col_names(dataframe, cat_th=10, car_th=20):
    """Identify categorical, numerical, and cardinal variables."""
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = [col for col in cat_cols + num_but_cat if col not in cat_but_car]
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O" and col not in num_but_cat]
    return cat_cols, num_cols, cat_but_car


def correlation_heatmap(df: pd.DataFrame, num_cols):
    """Show correlation matrix for numerical columns."""
    plt.figure(figsize=(12, 8))
    sns.heatmap(df[num_cols].corr(), annot=True, cmap="magma")
    plt.title("Correlation Matrix")
    plt.show()
