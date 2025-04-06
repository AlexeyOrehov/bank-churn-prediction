# Bank Churn Prediction Project

# 1. Importing libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
import shap

# 2. Load and preview dataset
df = pd.read_csv('churn.csv')
df.head()

# 3. EDA: checking for missing values
print(df.isnull().sum())

# 4. Target distribution
sns.countplot(data=df, x='Exited')
plt.title('Target variable distribution')
plt.show()

# 5. Feature analysis (example for Age)
sns.histplot(data=df, x='Age', hue='Exited', kde=True, bins=30)
plt.title('Age vs Exited')
plt.show()

# 6. Encoding categorical variables
df = pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=True)

# 7. Feature selection
X = df.drop(['RowNumber', 'CustomerId', 'Surname', 'Exited'], axis=1)
y = df['Exited']

# 8. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

# 9. Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_rf))

# 10. XGBoost model
xgb_model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,
    scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

print("\nXGBoost Classification Report:")
print(classification_report(y_test, y_pred_xgb))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_xgb))

# 11. SHAP feature importance
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)

# 12. Optional: SHAP dependence plot
shap.dependence_plot("Age", shap_values, X_test)

# 13. Conclusion
print("""
Key Insights:
- Age is the most influential feature for predicting churn.
- Clients from Germany and inactive users tend to churn more.
- XGBoost performs slightly better than RandomForest.
- SHAP provides explainable insights into model decisions.

This model can help banks to proactively engage with customers at high risk of churn.
""")
