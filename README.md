## Telco Customer Churn - Feature Engineering Tutorial

This project demonstrates **feature engineering and preprocessing** steps for the **Telco Customer Churn** dataset.  
The goal is to prepare the data for a predictive churn model by handling missing values, encoding categorical features, and creating new meaningful variables.

---

## Project Overview

**Business problem:**  
A telecommunications company wants to identify customers likely to churn (cancel their service).  
Before modeling, we perform **data analysis and feature engineering** to improve model performance.

**Dataset summary:**
- 7043 customers  
- 21 features (demographics, services, and account information)  
- Target variable: `Churn` (1 = customer left, 0 = stayed)

---

## Project Structure

telco-feature-engineering/
│
├── data/
│ └── telco_churn.csv
│
├── src/
│ ├── load_data.py # Load and clean dataset
│ ├── eda.py # Exploratory data analysis utilities
│ ├── feature_engineering.py # Missing/outlier handling + feature creation
│ ├── encoding.py # Label & one-hot encoding
│ └── modeling.py # Simple CatBoost model training
│
├── notebooks/
│ └── 01_telco_feature_engineering_walkthrough.py # Step-by-step analysis
│
├── README.md
└── requirements.txt

---

## Feature Engineering Steps

### Data Cleaning
- Convert `TotalCharges` to numeric.
- Fill missing values with the median.
- Cap outliers based on IQR thresholds.

### New Feature Creation
Some of the **custom engineered features** include:
- `NEW_TENURE_YEAR` → customer tenure grouped by year ranges  
- `NEW_Engaged` → contract duration flag (1 if ≥ 1 year)  
- `NEW_noProt` → customers without protection or backup services  
- `NEW_Young_Not_Engaged` → young + month-to-month contract  
- `NEW_TotalServices` → count of total subscribed services  
- `NEW_FLAG_ANY_STREAMING` → flag for streaming users  
- `NEW_FLAG_AutoPayment` → automatic payment flag  
- `NEW_AVG_Charges` → average monthly charge  
- `NEW_Increase` → price increase ratio  
- `NEW_AVG_Service_Fee` → cost per subscribed service

---

## Tech Stack

- **Python 3.10+**
- **Pandas**, **NumPy**, **Seaborn**, **Matplotlib**
- **Scikit-learn** for preprocessing & metrics
- **CatBoost** for model training

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/telco-feature-engineering.git
   cd telco-feature-engineering

2. Install dependencies:
   pip install -r requirements.txt

3. Place the dataset under /data folder:
   data/telco_churn.csv

4.Run the walkthrough notebook:
  python notebooks/01_telco_feature_engineering_walkthrough.py

##  Model Output Example

After training the CatBoost model, you’ll see performance metrics like:

Metric	Value
Accuracy	0.80
Recall	0.66
Precision	0.51
F1 Score	0.58
AUC	0.75

