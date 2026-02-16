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
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ===========================
# Load Model
# ===========================
model = pickle.load(open("model.pkl", "rb"))

# ===========================
# Page Configuration
# ===========================
st.set_page_config(page_title="Student Performance Dashboard", layout="wide")

# ===========================
# Custom CSS
# ===========================
st.markdown("""
<style>
body {
    background-color: #f0f2f6;
}
h1 {
    color: #1F4E79;
    font-size: 36px;
    font-weight: bold;
}
.card {
    background-color: #ADD8E6;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# ===========================
# Title
# ===========================
st.title("?? Student Performance Prediction Dashboard")

# ===========================
# Sidebar Inputs
# ===========================
st.sidebar.header("Input Student Data")
study_hours = st.sidebar.number_input("Study Hours", min_value=0, max_value=20, value=5)
attendance = st.sidebar.number_input("Attendance (%)", min_value=0, max_value=100, value=80)
assignment_score = st.sidebar.number_input("Assignment Score", min_value=0, max_value=100, value=75)
midterm_score = st.sidebar.number_input("Midterm Score", min_value=0, max_value=100, value=70)

# ===========================
# Tabs for Dashboard
# ===========================
tab1, tab2, tab3 = st.tabs(["Prediction", "Charts", "Summary"])

# --------- TAB 1: Prediction ---------
with tab1:
    st.subheader("?? Predicted Final Score")
    input_data = np.array([[study_hours, attendance, assignment_score, midterm_score]])
    prediction = model.predict(input_data)
    st.markdown(f"""
    <div class="card">
    <h2>Predicted Final Score: {prediction[0]:.2f}</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Risk Analysis
    if prediction[0] < 50:
        st.warning("? Student is at risk of failing")
    elif prediction[0] < 70:
        st.info("? Student is below average")
    else:
        st.success("? Student is performing well")

# --------- TAB 2: Charts ---------
with tab2:
    st.subheader("?? Feature Correlation")
    
    # Dummy dataset for visualization (replace with your dataset)
    data = pd.DataFrame({
        "study_hours": np.random.randint(1,10,100),
        "attendance": np.random.randint(50,100,100),
        "assignment_score": np.random.randint(40,100,100),
        "midterm_score": np.random.randint(30,100,100),
        "final_score": np.random.randint(40,100,100)
    })
    
    fig, ax = plt.subplots()
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
    
    st.subheader("?? Feature Distributions")
    fig2, ax2 = plt.subplots()
    data[['study_hours','attendance','assignment_score','midterm_score']].hist(ax=ax2, figsize=(10,4))
    st.pyplot(fig2)

# --------- TAB 3: Summary ---------
with tab3:
    st.subheader("?? Summary Statistics")
    st.dataframe(data.describe())
    
    st.markdown("**Average Final Score:** {:.2f}".format(data['final_score'].mean()))
    st.markdown("**Highest Final Score:** {:.2f}".format(data['final_score'].max()))
    st.markdown("**Lowest Final Score:** {:.2f}".format(data['final_score'].min()))
