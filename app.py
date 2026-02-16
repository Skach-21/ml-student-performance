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


import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import os

# ===========================
# Setup
# ===========================
st.set_page_config(page_title="Academy Dashboard", layout="wide", page_icon="??")

# ===========================
# Load or Initialize Data
# ===========================
DATA_FILE = "students.csv"

if os.path.exists(DATA_FILE):
    df = pd.read_csv(DATA_FILE)
else:
    # Create empty dataframe if CSV doesn't exist
    df = pd.DataFrame(columns=['StudentID','Name','StudyHours','Attendance','AssignmentScore','MidtermScore','FinalScore'])
    df.to_csv(DATA_FILE, index=False)

# Load ML model
model = pickle.load(open("model.pkl", "rb"))

# ===========================
# Login System
# ===========================
st.sidebar.title("?? Login")
# For demo, a simple dictionary
users = {"admin": "admin123", "teacher": "teach2026"}
username = st.sidebar.text_input("Username")
password = st.sidebar.text_input("Password", type="password")

login = st.sidebar.button("Login")
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if login:
    if username in users and password == users[username]:
        st.session_state.logged_in = True
        st.success(f"Welcome {username}!")
    else:
        st.error("? Invalid username or password")

# Stop app if not logged in
if not st.session_state.logged_in:
    st.stop()

# ===========================
# Sidebar Navigation
# ===========================
menu = st.sidebar.radio("Menu", ["Dashboard","Add Student","Update Student","Delete Student","Predict Final Score"])

# ===========================
# DASHBOARD
# ===========================
if menu == "Dashboard":
    st.title("?? Academy Dashboard")
    
    # KPIs
    total_students = df.shape[0]
    avg_score = df['FinalScore'].mean() if total_students>0 else 0
    max_score = df['FinalScore'].max() if total_students>0 else 0
    min_score = df['FinalScore'].min() if total_students>0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Students", total_students)
    col2.metric("Average Score", f"{avg_score:.2f}")
    col3.metric("Highest Score", max_score)
    col4.metric("Lowest Score", min_score)
    
    # Charts
    st.subheader("?? Feature Correlation")
    if total_students>0:
        fig = px.imshow(df[['StudyHours','Attendance','AssignmentScore','MidtermScore','FinalScore']].corr(), text_auto=True)
        st.plotly_chart(fig)
    
    st.subheader("?? Student Scores")
    if total_students>0:
        fig2 = px.bar(df, x='Name', y='FinalScore', color='FinalScore', color_continuous_scale='Viridis')
        st.plotly_chart(fig2)
    
    st.subheader("?? Student Data Table")
    st.dataframe(df)

# ===========================
# ADD STUDENT
# ===========================
elif menu == "Add Student":
    st.title("? Add New Student")
    with st.form("add_student_form"):
        student_id = st.text_input("Student ID")
        name = st.text_input("Name")
        study_hours = st.number_input("Study Hours", 0, 20, 5)
        attendance = st.number_input("Attendance (%)", 0, 100, 80)
        assignment_score = st.number_input("Assignment Score", 0, 100, 70)
        midterm_score = st.number_input("Midterm Score", 0, 100, 70)
        submitted = st.form_submit_button("Add Student")
        if submitted:
            final_score = model.predict(np.array([[study_hours, attendance, assignment_score, midterm_score]]))[0]
            new_row = {'StudentID': student_id, 'Name': name, 'StudyHours': study_hours,
                       'Attendance': attendance, 'AssignmentScore': assignment_score,
                       'MidtermScore': midterm_score, 'FinalScore': final_score}
            df = df.append(new_row, ignore_index=True)
            df.to_csv(DATA_FILE, index=False)
            st.success(f"? Student {name} added with predicted final score: {final_score:.2f}")

# ===========================
# UPDATE STUDENT
# ===========================
elif menu == "Update Student":
    st.title("?? Update Student")
    student_id = st.text_input("Enter Student ID to Update")
    if student_id:
        if student_id in df['StudentID'].values:
            student_row = df[df['StudentID']==student_id]
            st.write("Current Data:")
            st.dataframe(student_row)
            
            with st.form("update_student_form"):
                study_hours = st.number_input("Study Hours", 0, 20, int(student_row['StudyHours'].values[0]))
                attendance = st.number_input("Attendance (%)", 0, 100, int(student_row['Attendance'].values[0]))
                assignment_score = st.number_input("Assignment Score", 0, 100, int(student_row['AssignmentScore'].values[0]))
                midterm_score = st.number_input("Midterm Score", 0, 100, int(student_row['MidtermScore'].values[0]))
                submitted = st.form_submit_button("Update Student")
                if submitted:
                    final_score = model.predict(np.array([[study_hours, attendance, assignment_score, midterm_score]]))[0]
                    df.loc[df['StudentID']==student_id, ['StudyHours','Attendance','AssignmentScore','MidtermScore','FinalScore']] = \
                        [study_hours, attendance, assignment_score, midterm_score, final_score]
                    df.to_csv(DATA_FILE, index=False)
                    st.success(f"? Student {student_id} updated with new final score: {final_score:.2f}")
        else:
            st.error("? Student ID not found")

# ===========================
# DELETE STUDENT
# ===========================
elif menu == "Delete Student":
    st.title("?? Delete Student")
    student_id = st.text_input("Enter Student ID to Delete")
    if st.button("Delete"):
        if student_id in df['StudentID'].values:
            df = df[df['StudentID'] != student_id]
            df.to_csv(DATA_FILE, index=False)
            st.success(f"? Student {student_id} deleted")
        else:
            st.error("? Student ID not found")

# ===========================
# PREDICT FINAL SCORE
# ===========================
elif menu == "Predict Final Score":
    st.title("?? Predict Student Final Score")
    study_hours = st.number_input("Study Hours", 0, 20, 5)
    attendance = st.number_input("Attendance (%)", 0, 100, 80)
    assignment_score = st.number_input("Assignment Score", 0, 100, 70)
    midterm_score = st.number_input("Midterm Score", 0, 100, 70)
    if st.button("Predict"):
        final_score = model.predict(np.array([[study_hours, attendance, assignment_score, midterm_score]]))[0]
        if final_score < 50:
            st.error(f"? Predicted Final Score: {final_score:.2f} - At Risk")
        elif final_score < 70:
            st.warning(f"? Predicted Final Score: {final_score:.2f} - Below Average")
        else:
            st.success(f"? Predicted Final Score: {final_score:.2f} - Good Performance")
