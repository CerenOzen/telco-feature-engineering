"""
Telco Customer Churn - Feature Engineering Walkthrough

Goal:
- Explore the dataset
- Handle missing & outlier values
- Create meaningful new features
- Encode categorical variables
- Train a baseline CatBoost model
"""

import pandas as pd
from src.load_data import load_telco_data
from src.eda import check_df, grab_col_names, correlation_heatmap
from src.feature_engineering import handle_missing, replace_outliers, create_features
from src.encoding import label_encode_binary, one_hot_encode
from src.modeling import train_catboost

# 1. Load data
df = load_telco_data("data/telco_churn.csv")

# 2. Quick look
check_df(df)
cat_cols, num_cols, cat_but_car = grab_col_names(df)

# 3. Handle missing and outliers
df = handle_missing(df)
for col in num_cols:
    df = replace_outliers(df, col)

# 4. Feature creation
df = create_features(df)

# 5. Encoding
df = label_encode_binary(df)
df = one_hot_encode(df, exclude_cols=["Churn"])

# 6. Model training
model, metrics = train_catboost(df)
print("Model Performance:")
for k, v in metrics.items():
    print(f"{k}: {v:.2f}")
