# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pickle

loaded_model = pickle.load(open('C:/Users/KIIT0001/Desktop/MAHASWETA/ML/PROJECTS/trained_model.sav','rb')) 

input_data = (10,115,0,0,0,35.3,0.134,29)

numpy_array_of_input_data = np.asarray(input_data)

input_data_reshaped = numpy_array_of_input_data.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if(prediction[0] == '0'):
    print("Patient is diabetic")
else:
    print("Patient is not diabetic")