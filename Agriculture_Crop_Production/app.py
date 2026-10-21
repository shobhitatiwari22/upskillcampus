
import streamlit as st
import pandas as pd
import pickle
import numpy as np

st.set_page_config(page_title="Indian Crop Yield Predictor", page_icon="🌾", layout="centered")

st.title("🌾 Indian Crop Yield Prediction App")
st.write("This interactive application predicts regional agricultural crop yields across India based on your specific cultivation input metrics.")

@st.cache_resource
def load_assets():
    with open("Agriculture_Crop_Production/crop_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    with open("Agriculture_Crop_Production/model_columns.pkl", "rb") as columns_file:
        model_columns = pickle.load(columns_file)
    return model, model_columns

try:
    model, model_columns = load_assets()
except FileNotFoundError:
    st.error("⚠️ Model files not found! Please check that 'crop_model.pk1' and 'model_columns.pk1' are in the same folder.")
    st.stop()

crops = ['ARHAR', 'COTTON', 'GRAM', 'GROUNDNUT', 'MAIZE', 'MOONG', 'PADDY', 'RAPESEED AND MUSTARD', 'SUGARCANE', 'WHEAT']
states = ['Uttar Pradesh', 'Karnataka', 'Gujarat', 'Andhra Pradesh', 'Maharashtra', 'Punjab', 'Haryana', 'Rajasthan', 'Madhya Pradesh', 'Tamil Nadu', 'Bihar', 'Orissa', 'West Bengal']


st.subheader("📋 Enter Prediction Parameters")

col1, col2 = st.columns(2)
with col1:
    selected_crop = st.selectbox("Select Crop Type:", sorted(crops))
    cost_c2 = st.number_input("Cost of Cultivation C2 (Rs./Hectare):", min_value=0.0, value=25000.0, step=500.0)

with col2:
    selected_state = st.selectbox("Select Indian State:", sorted(states))
    cost_prod_c2 = st.number_input("Cost of Production C2 (Rs./Quintal):", min_value=0.0, value=2000.0, step=50.0)

if st.button("🔮 Predict Expected Crop Yield"):
    # Generate a baseline input row filled with zeros matching your original training columns
    input_data = pd.DataFrame(0, index=[0], columns=model_columns)

    # Populate numerical variables
    if 'Cost_of_Cultivation_C2' in input_data.columns:
        input_data['Cost_of_Cultivation_C2'] = cost_c2
    if 'Cost_of_Production_C2' in input_data.columns:
        input_data['Cost_of_Production_C2'] = cost_prod_c2

    # Populate One-Hot Encoded variables (matching dummy structures drop_first dynamic)
    crop_col = f"Crop_{selected_crop}"
    state_col = f"State_{selected_state}"

    if crop_col in input_data.columns:
        input_data[crop_col] = 1
    if state_col in input_data.columns:
        input_data[state_col] = 1

    # Execute model prediction
    prediction = model.predict(input_data)[0]

    # Display the result to the user
    st.success(f"### 🌱 Estimated Crop Yield: **{prediction:.2f} Quintals per Hectare**")
