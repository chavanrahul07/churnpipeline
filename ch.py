# Cell 1: Integrated Pipeline Setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, f1_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
print("Libraries imported and environment configured.")

# Cell 2: Step 1 - Create Medallion Folder Structure
folders = ['data/bronze', 'data/silver', 'data/gold', 'data/feature_store']
for folder in folders:
    os.makedirs(folder, exist_ok=True)
print("Medallion directory structure initialized.")

# Cell 3: Step 2 - Bronze Layer (Ingestion)
# Note: Ensure 'WA_Fn-UseC_-Telco-Customer-Churn.csv' is in your environment
input_file = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
if os.path.exists(input_file):
    raw_df = pd.read_csv(input_file)
    raw_df.to_csv('data/bronze/telco_raw.csv', index=False)
    print(f"Bronze Layer: Successfully ingested {len(raw_df)} records.")
else:
    print("Error: Input CSV not found. Please upload the Telco dataset.")

# Cell 4: Step 3 - Silver Layer (Initial Cleaning)
df_silver = pd.read_csv('data/bronze/telco_raw.csv')

# Handling 'TotalCharges' string-to-numeric conversion
df_silver['TotalCharges'] = pd.to_numeric(df_silver['TotalCharges'], errors='coerce')
df_silver = df_silver.dropna(subset=['TotalCharges'])

# Removing unnecessary ID column
df_silver.drop("customerID", axis=1, inplace=True)
df_silver.drop_duplicates(inplace=True)

df_silver.to_csv('data/silver/telco_cleaned.csv', index=False)
print("Silver Layer: Data cleaned (TotalCharges fixed, duplicates removed).")

# ... [Cells 5-30: Descriptive Statistics & Detailed EDA from the Churn notebook] ...
# Includes distribution plots, pie charts for churn balance, and correlation heatmaps.

# Cell 31: Data Distribution Visualization (Integrated EDA)
def plot_dist(df, col):
    plt.figure(figsize=(8,4))
    sns.histplot(df[col], kde=True, color='blue')
    plt.title(f'Distribution of {col}')
    plt.show()

plot_dist(df_silver, 'tenure')
plot_dist(df_silver, 'MonthlyCharges')

# Cell 32: Step 4 - Gold Layer (Feature Engineering)
df_gold = pd.read_csv('data/silver/telco_cleaned.csv')

# Feature Engineering: Grouping Tenure
def tenure_group(tenure):
    if tenure <= 12: return 'Short-term'
    elif tenure <= 36: return 'Medium-term'
    else: return 'Long-term'

df_gold['TenurePeriod'] = df_gold['tenure'].apply(tenure_group)

# Encoding Categorical Variables
le = LabelEncoder()
for col in df_gold.select_dtypes(include='object').columns:
    df_gold[col] = le.fit_transform(df_gold[col])

df_gold.to_csv('data/gold/telco_features.csv', index=False)
print("Gold Layer: Feature engineering complete. Data is now model-ready.")

# Cell 33: Step 5 - Feature Store Versioning
feature_set = pd.read_csv('data/gold/telco_features.csv')
feature_set['Pipeline_Version'] = "v1.0"
feature_set['Processed_At'] = datetime.now()
feature_set.to_csv('data/feature_store/final_training_v1.csv', index=False)
print("Feature Store: Versioned dataset saved.")

# ... [Cells 34-80: Advanced ML Modeling & Comparisons] ...
# This includes the logic for Train/Test split, Scaling, 
# and the training of Logistic Regression, Random Forest, and XGBoost.

# Cell 81: Model Training - XGBoost
X = feature_set.drop(['Churn', 'Pipeline_Version', 'Processed_At'], axis=1)
y = feature_set['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

xgb_model = XGBClassifier(eval_metric='logloss')
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)

# Cell 82: Evaluation - Classification Report
print(classification_report(y_test, y_pred))

# Cell 83: Visualization - Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
plt.title('Confusion Matrix - Churn Integrated Model')
plt.show()

# Cell 84: Visualization - ROC Curve
fpr, tpr, _ = roc_curve(y_test, xgb_model.predict_proba(X_test)[:,1])
plt.plot(fpr, tpr, label='XGBoost')
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend()
plt.show()

# ... [Cells 85-89: Conclusion and Save Model] ...
