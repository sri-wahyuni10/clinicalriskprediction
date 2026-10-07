import numpy as np
import pandas as pd
from xgboost import XGBClassifier 
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib
import shap
import matplotlib.pyplot as plt

header_cols = column_names = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"
]
df =  pd.read_csv('./data/raw/processed.cleveland.data', names=header_cols)

df = df.replace("?", np.nan)
df = df.dropna()

cols = ['ca', 'thal']
for col in cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    
#print(df.dtypes)

df['num'] = df['num'].clip(upper=1)

print('nilai unik thal:', df['thal'].unique())
X = df.drop('num', axis=1)
y = df['num']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#print(X.dtypes)

cols_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
scaler = StandardScaler()
X_train[cols_scale] = scaler.fit_transform(X_train[cols_scale])

X_test[cols_scale] = scaler.transform(X_test[cols_scale])

model = LogisticRegression()
model.fit(X_train, y_train)

#print("===Model Regression=== ")
y_pred = model.predict(X_test)
classification_rep = classification_report(y_test, y_pred)
#print("Classification Report:")
#print(classification_rep)


cm = confusion_matrix(y_test, y_pred)
#print("Confusion Matrix:")
#print(cm)
'''
#print("===Model Random Forest===")
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)
classification_rep_rf = classification_report(y_test, y_pred_rf)

#print("Classification Report (Random Forest):")
#print(classification_rep_rf)

print("===Model XGBoost===")
xgb_model =XGBClassifier( eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
classification_rep_xgb = classification_report(y_test, y_pred_xgb)
print("Classification Report (XGBoost):")
print(classification_rep_xgb)
print("Confusion Matrix (XGBoost):")
cm_xgb = confusion_matrix(y_test, y_pred_xgb)
print(cm_xgb)
'''
#saving model regression
joblib.dump(model, 'models/logistic_regression_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(X_train, 'models/X_train.pkl')

explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)
print("SHAP Values:")
print(shap_values.shape)

#shap.summary_plot(shap_values, X_test)
#plt.tight_layout()
#plt.savefig('./data/processed/shap_summary_plot.png')

shap.plots.waterfall(shap_values[0])
