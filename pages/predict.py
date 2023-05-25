import streamlit as st
import numpy as np
from pickle import load
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import os

sc = load(open('model/standard_scaler.pkl', 'rb'))
svm_model = load(open('model/svm_model.pkl', 'rb'))
#Page Heading
st.header(":blue[Diabetes Prediction]")

col1, col2, col3= st.columns(3)
with col1:
  Pregnancies = st.text_input("Pregnancies", placeholder="Enter the value")
   
with col2:
   Glucose = st.text_input("Glucose", placeholder="Enter the value")

with col3:
   BloodPressure = st.text_input("BloodPressure", placeholder="Enter the value")

col1, col2,col3= st.columns(3)
with col1:
   SkinThickness = st.text_input("SkinThickness", placeholder="Enter the value")

with col2:
   Insulin = st.text_input("Insulin", placeholder="Enter the value")


with col3:
   BMI= st.text_input("BMI", placeholder="Enter the value")

col1, col2,= st.columns(2)
with col1:
   DiabetesPedigreeFunction = st.text_input("DiabetesPedigreeFunction", placeholder="Enter the value")


with col2:
   Age = st.text_input("Age", placeholder="Enter the value")


btn_click = st.button("Predict")

if btn_click == True:
    if Pregnancies and Glucose and BloodPressure and SkinThickness and Insulin and BMI and DiabetesPedigreeFunction and Age:
        query_point = np.array([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]).reshape(1, -1)
        query_point_transformed = sc.transform(query_point)
        pred = svm_model.predict(query_point_transformed)
        if pred[0] == 1:
            st.markdown("<h2 style='color: red;'>You have diabetes.</h2>", unsafe_allow_html=True)
            
        else:
            st.markdown("<h2 style='color: blue;'>You don't have diabetes.</h2>",unsafe_allow_html=True)
            
    else:
        st.error("Enter the values properly.")
