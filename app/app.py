import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(
    page_title="HeartCare AI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)


col_left, col_right = st.columns([2, 1])



with col_left:

    st.title("Clinical Risk Prediction App")
    st.write("This app predicts the clinical risk based on patient data using a trained machine learning model.")
    st.write("Please enter the patient data below:")
    
    with st.container(border=True):

        with st.form(key='patient_data_form'):
            age = st.number_input('Age (Umur)', min_value=0, max_value=120,value=25)
            sex = st.selectbox('Sex (Female = 0, Male = 1)', options=[0,1])
            cp = st.selectbox('Chest Pain Type (Tipe Nyeri Dada)', options=[1,2,3,4])
            trestbps = st.number_input('Resting Blood Pressure (Tekanan Darah Istirahat)', min_value=0, max_value=300,value=120)
            chol = st.number_input('Serum Cholesterol (Kolesterol Serum)', min_value=0, max_value=600,value=200)
            fbs = st.selectbox('Fasting Blood Sugar (Gula Darah Puasa)', options=[0,1])
            restecg = st.selectbox('Resting Electrocardiographic Results (Hasil Elektrokardiografi Istirahat)', options=[0,1,2])
            thalach = st.number_input('Maximum Heart Rate Achieved (Detak Jantung Maksimum yang Dicapai)', min_value=0, max_value=250,value=150)
            exang = st.selectbox('Exercise Induced Angina (Angina yang Dipicu oleh Olahraga)', options=[0,1])
            oldpeak = st.number_input('ST Depression Induced by Exercise Relative to Rest (Depresi ST yang Dipicu oleh Olahraga [0.0 - 7.0])', min_value=0.0, max_value=7.0,value=1.0)
            slope = st.selectbox('Slope of the Peak Exercise ST Segment (Kemiringan Segmen ST Latihan Puncak)', options=[1,2,3])
            ca = st.selectbox('Number of Major Vessels (0-3) Colored by Fluoroscopy (Jumlah Pembuluh Darah Utama (0-3) yang Diwarnai oleh Fluoroskopi)', options=[0,1,2,3])
            thal = st.selectbox('Thalassemia (Talassemia)', options=[3, 6, 7])
            
            submit_button = st.form_submit_button(label='Predict')
        
with col_right:
    st.write("Hasil prediksi")
    if submit_button:
        input_data = {
            "age": age,
            "sex": sex,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalach": thalach,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal
            }
            
        #response = requests.post("http://localhost:8000/predict", json=input_data)
        response = requests.post("http://api:8000/predict", json=input_data)
            
        if response.status_code == 200:
            result = response.json()
            prediction = result['prediction']
            probability = result['probability']
            if prediction == 1:
                st.error("The patient is at risk. Please consult a healthcare professional.")
            else :
                st.success("The patient is not at risk. However, regular check-ups are recommended.")
            with st.container(border=True):
                st.metric(label="Probability of Risk", value=f"{probability:.2%}")
            with st.container(border=True):
                st.write("SHAP Values:")
                shap_values = pd.Series(result['shap_values'])
                shap_sorted = shap_values.reindex(shap_values.abs().sort_values(ascending=True).index)
                colors = ['crimson' if val > 0 else 'teal' for val in shap_sorted.values]
                fig = go.Figure(go.Bar(
                    x=shap_sorted.values,
                    y=shap_sorted.index,
                    orientation='h',
                    marker_color=colors
                ))
                fig.update_layout(title="Faktor yang Mempengaruhi Prediksi", xaxis_title="SHAP Value")
                st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.error("Error in prediction. Please try again.")                 
    
    
