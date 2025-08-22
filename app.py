# app.py — Customer Salary Prediction (ANN)

import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

st.set_page_config(page_title="Salary Predictor", page_icon="💼", layout="centered")
st.title("💼 Customer Salary Prediction (ANN)")

# ---------- Load artifacts (cached) ----------
@st.cache_resource
def load_artifacts():
    # Load model
    try:
        model = tf.keras.models.load_model("salary_model.h5")
    except Exception as e:
        st.error("Couldn't load model file `salary_model.h5`. Make sure it's in the app folder.")
        st.stop()

    # Load scaler
    try:
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
    except Exception:
        st.error("Couldn't load `scaler.pkl`.")
        st.stop()

    # Load encoders
    try:
        with open("label_encoder_gender.pkl", "rb") as f:
            le_gender = pickle.load(f)
    except Exception:
        st.error("Couldn't load `label_encoder_gender.pkl`.")
        st.stop()

    try:
        with open("onehot_encoder_geo.pkl", "rb") as f:
            ohe_geo = pickle.load(f)
    except Exception:
        st.error("Couldn't load `onehot_encoder_geo.pkl`.")
        st.stop()

    # Load training column order
    try:
        with open("X_columns.pkl", "rb") as f:
            X_columns = pickle.load(f)
    except Exception:
        st.error("Couldn't load `X_columns.pkl`. Save your training feature columns during training.")
        st.stop()

    return model, scaler, le_gender, ohe_geo, X_columns

model, scaler, le_gender, ohe_geo, X_columns = load_artifacts()

# Geography choices from the actual trained encoder
try:
    GEO_CHOICES = list(ohe_geo.categories_[0])
except Exception:
    GEO_CHOICES = ["France", "Germany", "Spain"]  # fallback

# ---------- UI ----------
st.subheader("Enter Customer Details")
with st.form("predict_form", clear_on_submit=False):
    col1, col2 = st.columns(2)
    with col1:
        customer_id = st.number_input("CustomerId (optional)", min_value=0, value=15634602, step=1)
        credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650, step=1)
        age = st.number_input("Age", min_value=18, max_value=100, value=40, step=1)
        tenure = st.number_input("Tenure (years)", min_value=0, max_value=10, value=3, step=1)
        balance = st.number_input("Balance", min_value=0.0, value=60000.0, step=100.0, format="%.2f")
    with col2:
        num_products = st.number_input("Num Of Products", min_value=1, max_value=4, value=2, step=1)
        has_cr_card = st.checkbox("Has Credit Card", value=True)
        is_active_member = st.checkbox("Is Active Member", value=True)
        exited = st.selectbox("Exited (0 = No, 1 = Yes)", options=[0, 1], index=0)
        gender = st.selectbox("Gender", options=["Male", "Female"], index=0)
        geography = st.selectbox("Geography", options=GEO_CHOICES, index=0)

    submitted = st.form_submit_button("Predict Salary 💡")

# ---------- Predict ----------
if submitted:
    try:
        # Build one-row DataFrame with raw inputs
        input_data = pd.DataFrame([{
            "CustomerId": int(customer_id),
            "CreditScore": int(credit_score),
            "Gender": gender,               # will be label-encoded
            "Age": int(age),
            "Tenure": int(tenure),
            "Balance": float(balance),
            "NumOfProducts": int(num_products),
            "HasCrCard": int(has_cr_card),
            "IsActiveMember": int(is_active_member),
            "Exited": int(exited),
            "Geography": geography          # will be one-hot encoded
        }])

        # Encode Gender
        input_data["Gender"] = le_gender.transform(input_data["Gender"])

        # One-hot encode Geography (using the exact training encoder)
        geo_arr = ohe_geo.transform(input_data[["Geography"]]).toarray()
        geo_cols = ohe_geo.get_feature_names_out(["Geography"])
        geo_df = pd.DataFrame(geo_arr, columns=geo_cols, index=input_data.index)

        # Combine and drop original Geography
        input_data = pd.concat([input_data.drop(columns=["Geography"]), geo_df], axis=1)

        # Reindex to match training columns (order + any missing)
        input_data = input_data.reindex(columns=X_columns, fill_value=0)

        # Scale
        X_scaled = scaler.transform(input_data)

        # Predict once (avoid retracing spam)
        y_pred = model.predict(X_scaled, verbose=0)
        pred_salary = float(y_pred[0, 0])

        st.success(f"💰 Predicted Estimated Salary: **{pred_salary:,.2f}**")

        with st.expander("See model inputs (after preprocessing)"):
            st.write("Feature order exactly as during training:")
            st.dataframe(input_data)

    except Exception as e:
        st.error("Prediction failed. Check your artifacts and inputs.")
        st.exception(e)



