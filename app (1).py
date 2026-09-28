
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(
    page_title="AI Traffic Congestion Predictor",
    page_icon="🚦"
)

st.title("🚦 AI Traffic Congestion Predictor")
st.write("Predict traffic congestion using Machine Learning.")

# Load dataset
df = pd.read_csv("traffic_data.csv")

X = df[
    ["Hour", "Day", "Vehicles", "Average_Speed", "Rainfall", "Accident"]
]
y = df["Congestion_Level"]

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
model.fit(X, y)

st.subheader("Enter Traffic Details")

hour = st.number_input("Hour", min_value=0, max_value=23, value=18)
day = st.number_input("Day", min_value=1, max_value=7, value=5)
vehicles = st.number_input("Number of Vehicles", min_value=0, value=500)
speed = st.number_input("Average Speed (km/h)", min_value=0, value=40)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, value=0.0)
accident = st.selectbox("Accident Reported?", [0, 1])

if st.button("Predict Traffic 🚦"):

    new_data = pd.DataFrame({
        "Hour": [hour],
        "Day": [day],
        "Vehicles": [vehicles],
        "Average_Speed": [speed],
        "Rainfall": [rainfall],
        "Accident": [accident]
    })

    prediction = model.predict(new_data)[0]

    traffic_levels = {
        0: "🟢 Low Traffic",
        1: "🟡 Medium Traffic",
        2: "🟠 High Traffic",
        3: "🔴 Critical Traffic"
    }

    result = traffic_levels.get(prediction, "Unknown")

    st.success(f"Predicted Traffic Level: {result}")
