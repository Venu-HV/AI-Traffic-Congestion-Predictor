
import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓"
)

st.title("🎓 Student Performance Prediction")
st.write("Predict a student's final marks using academic performance data.")

# Load dataset
url = "https://raw.githubusercontent.com/Venu-HV/student-performance-prediction/main/student_performance.csv"
df = pd.read_csv(url)

# Features and target
X = df[["Study_Hours", "Attendance", "Previous_Marks", "Assignment_Score"]]
y = df["Final_Marks"]

# Train model
model = LinearRegression()
model.fit(X, y)

st.subheader("Enter Student Details")

study_hours = st.number_input("Study Hours", min_value=0, max_value=24, value=6)
attendance = st.number_input("Attendance (%)", min_value=0, max_value=100, value=85)
previous_marks = st.number_input("Previous Marks", min_value=0, max_value=100, value=75)
assignment_score = st.number_input("Assignment Score", min_value=0, max_value=100, value=80)

if st.button("Predict Final Marks"):

    new_student = pd.DataFrame({
        "Study_Hours": [study_hours],
        "Attendance": [attendance],
        "Previous_Marks": [previous_marks],
        "Assignment_Score": [assignment_score]
    })

    prediction = model.predict(new_student)[0]

    st.success(f"Predicted Final Marks: {prediction:.2f}")
