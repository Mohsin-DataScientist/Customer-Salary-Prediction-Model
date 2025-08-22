import streamlit as st
import tensorflow as tf
import pickle
import pandas as pd

# ==============================
# Cache model & preprocessing tools
# ==============================
@st.cache_resource
def load_model_and_tools():
    model = tf.keras.models.load_model("salary_model.h5")

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("label_encoder_gender.pkl", "rb") as f:
        le_gender = pickle.load(f)
    with open("onehot_encoder_geo.pkl", "rb") as f:
        ohe_geo = pickle.load(f)
    with open("X_columns.pkl", "rb") as f:
        X_columns = pickle.load(f)

    return model, scaler, le_gender, ohe_geo, X_columns

model, scaler, le_gender, ohe_geo, X_columns = load_model_and_tools()

# ==============================
# UI
# ==============================
st.title("💰 Customer Salary Prediction (ANN Model)")
st.write("Enter customer details to predict their **Estimated Salary**.")

with st.form("salary_form"):
    credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=600)
    geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=18, max_value=100, value=40)
    tenure = st.number_input("Tenure (Years)", min_value=0, max_value=10, value=3)
    balance = st.number_input("Balance", min_value=0, value=60000)
    products = st.number_input("Number of Products", min_value=1, max_value=4, value=2)
    has_card = st.selectbox("Has Credit Card", [0, 1])
    active_member = st.selectbox("Is Active Member", [0, 1])

    submitted = st.form_submit_button("Predict Salary")

if submitted:
    # Prepare input
    input_data = pd.DataFrame([{
        "CreditScore": credit_score,
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": products,
        "HasCrCard": has_card,
        "IsActiveMember": active_member,
        "EstimatedSalary": 0  # dummy column dropped later
    }])

    # Encode Gender
    input_data["Gender"] = le_gender.transform(input_data["Gender"])

    # Encode Geography
    geo_encoded = ohe_geo.transform(input_data[["Geography"]]).toarray()
    geo_encoded_df = pd.DataFrame(
        geo_encoded, columns=ohe_geo.get_feature_names_out(["Geography"])
    )
    input_data = pd.concat([input_data.drop("Geography", axis=1), geo_encoded_df], axis=1)

    # Align column order
    input_data = input_data[X_columns]

    # Scale
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)
    predicted_salary = prediction[0][0]

    st.success(f"💵 Predicted Estimated Salary: **{predicted_salary:,.2f}**")
