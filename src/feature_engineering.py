import pandas as pd


def replace_outliers(df, col, q1=0.05, q3=0.95):
    """Cap outliers using IQR limits."""
    low = df[col].quantile(q1)
    high = df[col].quantile(q3)
    iqr = high - low
    lower = low - 1.5 * iqr
    upper = high + 1.5 * iqr
    df[col] = df[col].clip(lower, upper)
    return df


def handle_missing(df):
    """Fill missing TotalCharges with median."""
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
    return df


def create_features(df):
    """Create new meaningful features."""
    df["NEW_TENURE_YEAR"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 36, 48, 60, 72],
        labels=["0-1 Year", "1-2 Year", "2-3 Year", "3-4 Year", "4-5 Year", "5-6 Year"],
    )

    df["NEW_Engaged"] = df["Contract"].apply(lambda x: 1 if x in ["One year", "Two year"] else 0)
    df["NEW_noProt"] = df.apply(
        lambda x: 1
        if (x["OnlineBackup"] != "Yes")
        or (x["DeviceProtection"] != "Yes")
        or (x["TechSupport"] != "Yes")
        else 0,
        axis=1,
    )
    df["NEW_Young_Not_Engaged"] = df.apply(
        lambda x: 1 if (x["NEW_Engaged"] == 0) and (x["SeniorCitizen"] == 0) else 0, axis=1
    )

    df["NEW_TotalServices"] = (
        (df[["PhoneService", "InternetService", "OnlineSecurity", "OnlineBackup",
             "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]] == "Yes")
        .sum(axis=1)
    )

    df["NEW_FLAG_ANY_STREAMING"] = df.apply(
        lambda x: 1 if (x["StreamingTV"] == "Yes") or (x["StreamingMovies"] == "Yes") else 0,
        axis=1,
    )

    df["NEW_FLAG_AutoPayment"] = df["PaymentMethod"].apply(
        lambda x: 1 if x in ["Bank transfer (automatic)", "Credit card (automatic)"] else 0
    )

    df["NEW_AVG_Charges"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["NEW_Increase"] = df["NEW_AVG_Charges"] / df["MonthlyCharges"]
    df["NEW_AVG_Service_Fee"] = df["MonthlyCharges"] / (df["NEW_TotalServices"] + 1)

    return df
