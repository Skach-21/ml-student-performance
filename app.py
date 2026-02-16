import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("model.pkl", "rb"))

st.title("Student Performance Prediction")

study_hours = st.number_input("Study Hours")
attendance = st.number_input("Attendance (%)")
assignment_score = st.number_input("Assignment Score")
midterm_score = st.number_input("Midterm Score")

if st.button("Predict"):
    input_data = np.array([[study_hours, attendance, assignment_score, midterm_score]])
    prediction = model.predict(input_data)
    st.success(f"Predicted Final Score: {prediction[0]:.2f}")


