# Heart Price Prediction - Deployment

# Importing Necessary Packages
# For Deployment
import streamlit as st
from streamlit_option_menu import option_menu
# For Data Collection
import pandas as pd
# For Mathematical Processing
import numpy as np
# For Visualization
from matplotlib import pyplot as plt
import seaborn as sns
# For Model Selection
from sklearn.model_selection import train_test_split
# For Model Building
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

# Building Skeleton for Web App
selected = option_menu(
       menu_title = None ,#"Main Menu",
       options = ["Home","Model"],
       icons = ["house","robot"],
       menu_icon = "menu-button",
       default_index = 0,
       orientation = "horizontal",  #to make the menubar horizontal 
    )

# Creating outline for HOME
if selected == "Home":
    with st.container():
        st.title('Heart Disease Prediction')
        st.subheader('A classification based Machine Learning Project')
    
    with st.container():
        st.title('What this Heart Disease Prediction project all about?')
        st.write('Cardiovascular diseases, including heart disease, remain a significant global health concern and a leading cause of mortality. Early detection and prediction of heart disease can play a pivotal role in improving patient outcomes and reducing the burden on healthcare systems. This project aims to develop a robust and accurate heart disease prediction model using machine learning technique called classification.')
        st.subheader('This is a sample Dataset:')
        st.text("In the column target, 1 = Has heart disease & 0 = Does not have heart disease")
        df = pd.read_csv("heart_disease_data.csv")
        st.write(df)

# Creating outline for MODEL
if selected == "Model":
    st.title('Heart Disease Prediction')
    st.subheader('Enter the Details')
 
    df = pd.read_csv("heart_disease_data.csv")
    X = df.drop(columns='target', axis=1)
    Y = df['target']
    
    model = st.selectbox('Choose a Model',
      ('Logistic Regression', 'Random Forest Classifier', 'Naive Bayes Classifier'))
    
    if model =='Logistic Regression':
        m = LogisticRegression()
        m.fit(X, Y)
    elif model =='Random Forest Classifier':
        m = RandomForestClassifier(n_estimators=100)
        m.fit(X, Y)
    else:
        m = GaussianNB()
        m.fit(X, Y)
        
    # To return predicted results
    def hdp(input_data):
        na = np.asarray(input_data)
        d = na.reshape(1, -1)

        prediction = m.predict(d)
        print(prediction)

        if (prediction[0] == 0):
            return 'The patient does not have heart disease'
        else:
            return 'The patient has heart disease'
        
    # To get input from the user
    def main():
        age = st.text_input('Age')
        sex_1 = st.selectbox('Sex', ('Male', 'Female'))
        cp_1 = st.selectbox('Chest Pain Type', ('Typical Angina', 'Atypical Angina', 'Non Aanginal Pain', 'Asymptomatic'))
        trestbps = st.slider('Resting Blood Pressure', min_value=90, max_value=200, value=90, step=5)
        chol = st.slider('Serum Cholestoral in mg/dl', min_value=125, max_value=565, value=125, step=5)
        fbs_1 = st.selectbox('Whether the Fasting Blood Sugar is greater than 120 mg/dl?', ('Yes', 'No'))
        restecg_1 = st.selectbox('Resting Electrocardiographic Results', ('ASA Grade I', 'ASA Grade II', 'ASA Grade III'))
        thalach = st.slider('Maximum Heart Rate Achieved', min_value=70, max_value=205, value=70, step=5)
        exang_1 = st.selectbox('Does Exercise induced Angina?', ('Yes', 'No'))
        oldpeak = st.slider('Oldpeak', min_value=0.0, max_value=6.5, value=0.0, step=0.5)
        slope_1 = st.selectbox('Slope Type', ('Upsloping', 'Flat', 'Downsloping'))
        ca = st.selectbox('Number of Major Vessels', (0, 1, 2, 3))
        thal_1 = st.selectbox('Type of Thalassemia defect', ('Normal', 'Fixed Defect', 'Reversable Defect', 'Genetic Defect'))
        
        # for sex column
        if sex_1=='Male':
            sex = 1
        else:
            sex = 2
            
        # for cp column
        if cp_1=='Typical Angina':
            cp = 0
        elif cp_1=='Atypical Angina':
            cp = 1
        elif cp_1=='Non Aanginal Pain':
            cp = 2
        else:
            cp = 3
            
        # for fbs column
        if fbs_1=='Yes':
            fbs = 1
        else:
            fbs = 0
            
        # for restecg column
        if restecg_1=='ASA Grade I':
            restecg = 0
        elif restecg_1=='ASA Grade II':
            restecg = 1
        else:
            restecg = 2
            
        # for exang column
        if exang_1=='Yes':
            exang = 1
        else:
            exang = 0
            
        # for slope column
        if slope_1=='Upsloping':
            slope = 0
        elif slope_1=='Flat':
            slope = 1
        else:
            slope = 2
            
        # for thal column
        if thal_1=='Normal':
            thal = 0
        elif thal_1=='Fixed Defect':
            thal = 1
        elif thal_1=='Reversable Defect':
            thal = 2
        else:
            thal = 3
            
        hd=''
        
        # To establish connection between buttons
        if st.button('Get Results'):
            hd = hdp([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal])
            st.success(hd)
            
