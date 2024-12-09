import streamlit as st
import numpy as np
import pickle

# Load the model
model = pickle.load(open('model.pkl', 'rb'))

# Prediction function
def prediction_result(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = model.predict(input_data_reshaped)

    if prediction[0] == 0:
        return 'The person is not diabetic.'
    else:
        return 'The person is diabetic.'

# Streamlit app
def main():
    # Set app title and description with some custom styling
    st.markdown("""
    <style>
    body {
        background-color: #f4f7fb;
        color: #333333;
        font-family: Arial, sans-serif;
    }
    .stButton>button {
        background-color: #007bff;
        color: white;
        font-size: 18px;
    }
    .stTextInput>div>input {
        border: 1px solid #007bff;
        border-radius: 5px;
        padding: 8px;
        font-size: 16px;
    }
    .stTextInput>label {
        color: #007bff;
    }
    .title {
        font-size: 30px;
        color: #4CAF50;
        text-align: center;
    }
    .subheader {
        font-size: 22px;
        color: #4CAF50;
        text-align: center;
    }
    .result {
        font-size: 20px;
        font-weight: bold;
        color: #333333;
        background-color: #e7f7e7;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
    }
    .info-box {
        background-color: #ffffff;
        border: 1px solid #007bff;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        margin-top: 30px;
    }
    .header {
        text-align: center;
        font-size: 40px;
        color: #0056b3;
    }
    .expander {
        background-color: #f0f8ff;
        border: 1px solid #4CAF50;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="header">Diabetes Prediction Web App</p>', unsafe_allow_html=True)
    st.markdown('<p class="subheader">Enter the details below to get predictions</p>', unsafe_allow_html=True)

    # Input fields using columns for better layout
    col1, col2 = st.columns(2)

    with col1:
        Pregnancies = st.text_input("Number of Pregnancies")
        Glucose = st.text_input("Glucose Level")
        BloodPressure = st.text_input("Blood Pressure Level")
        SkinThickness = st.text_input("Skin Thickness")

    with col2:
        Insulin = st.text_input("Insulin Level")
        BMI = st.text_input("BMI")
        DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function")
        Age = st.text_input("Age")

    diagnosis = ''

    # Button for prediction
    if st.button("Get Diabetes Prediction"):
        try:
            input_data = [
                float(Pregnancies),
                float(Glucose),
                float(BloodPressure),
                float(SkinThickness),
                float(Insulin),
                float(BMI),
                float(DiabetesPedigreeFunction),
                float(Age)
            ]
            diagnosis = prediction_result(input_data)
        except ValueError:
            diagnosis = "Please enter valid numeric values for all fields."

    # Display the result
    if diagnosis:
        st.markdown(f'<p class="result">{diagnosis}</p>', unsafe_allow_html=True)

    # Expander for additional information
    with st.expander("How Does It Work?", expanded=True):
        st.markdown("""
        This model predicts whether a person is diabetic or not based on various health parameters.
        The prediction is based on the following features:
        - Number of Pregnancies
        - Glucose Level
        - Blood Pressure
        - Skin Thickness
        - Insulin
        - BMI (Body Mass Index)
        - Diabetes Pedigree Function
        - Age
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
