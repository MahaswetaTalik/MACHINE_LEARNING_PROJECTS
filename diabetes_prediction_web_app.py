# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 22:10:21 2025

@author: KIIT0001
"""

import numpy as np
import pickle
import streamlit as st


loaded_model = pickle.load(open('C:/Users/KIIT0001/Desktop/MAHASWETA/ML/PROJECTS/trained_model.sav','rb')) 

# creating a function for prediction
def diabetes_prediction(input_data):
    input_data = (10,115,0,0,0,35.3,0.134,29)

    numpy_array_of_input_data = np.asarray(input_data)

    input_data_reshaped = numpy_array_of_input_data.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if(prediction[0] == '0'):
        return 'Patient is diabetic'
    else:
        return 'Patient is not diabetic'
    
    
def main():
    
    # giving a title
    
    st.title('Diabetes Prediction Web App')
    
    
    # getting the input data from the user

    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure value')
    SkinThickness = st.text_input('Skin Thickness value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    Age = st.text_input('Age of the Person')
    
    # code for prediction
    
    diagnosis = ''
    
    # creating a button for prediction
    
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,
                                         BMI,DiabetesPedigreeFunction,Age])
    
    st.success(diagnosis)
    
    
    
if __name__ == '__main__':
    main()

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    