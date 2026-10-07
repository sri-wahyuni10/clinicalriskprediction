from fastapi import FastAPI
import joblib
from pydantic import BaseModel
import pandas as pd
import shap

class PatientData(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float
    

model = joblib.load('models/logistic_regression_model.pkl')
scaler = joblib.load('models/scaler.pkl')
training_data = joblib.load('models/X_train.pkl')

explainer = shap.Explainer(model, training_data)

app = FastAPI()

@app.get("/")
def root():
    return {'message': 'hello world'}

@app.post("/predict")
def predict(data: PatientData):
    input_data = [[
        data.age,
        data.sex,
        data.cp,
        data.trestbps,
        data.chol,
        data.fbs,
        data.restecg,
        data.thalach,
        data.exang,
        data.oldpeak,
        data.slope,
        data.ca,
        data.thal
    ]]
    
    # Scale the input data
    cols_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    input_df = pd.DataFrame(input_data, columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'])
    input_df[cols_scale] = scaler.transform(input_df[cols_scale])
    shap_values = explainer(input_df)
    shap_dict = {key: float(value) for key, value in zip(input_df.columns, shap_values.values[0])}
    
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[:, 1]  # Probability of the positive class
    return {'prediction': int(prediction[0]), 'probability': float(probability[0]), 'shap_values': shap_dict}
