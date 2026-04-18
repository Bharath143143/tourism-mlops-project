
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from huggingface_hub import hf_hub_download

MODEL_REPO_ID = "Bharath1434/tourism-package-prediction-model"

st.set_page_config(
    page_title="Wellness Tourism Package Predictor",
    page_icon="🌿",
    layout="wide"
)

@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id=MODEL_REPO_ID,
        filename="best_xgb_model.pkl"
    )
    return joblib.load(model_path)

@st.cache_data
def load_feature_names():
    """Load feature names from JSON — no auth needed, no large CSV download."""
    try:
        features_path = hf_hub_download(
            repo_id=MODEL_REPO_ID,
            filename="feature_names.json"
        )
        with open(features_path, "r") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading feature names: {e}")
        st.stop()

model           = load_model()
expected_features = load_feature_names()

st.title("🌿 Wellness Tourism Package Predictor")
st.markdown("**Visit with Us** | Predict whether a customer will purchase the Wellness Package")
st.divider()

st.sidebar.header("📋 Customer Information")

age             = st.sidebar.slider("Age", 18, 61, 36)
monthly_income  = st.sidebar.number_input("Monthly Income (₹)", 5000, 98678, 22347)
num_trips       = st.sidebar.slider("Number of Trips/Year", 1, 22, 3)
duration_pitch  = st.sidebar.slider("Duration of Sales Pitch (min)", 5, 127, 15)
num_persons     = st.sidebar.slider("Persons Visiting", 1, 5, 3)
num_children    = st.sidebar.slider("Children (<5 yrs)", 0, 3, 0)
num_followups   = st.sidebar.slider("Number of Followups", 1, 6, 4)
pitch_score     = st.sidebar.slider("Pitch Satisfaction Score", 1, 5, 3)
property_star   = st.sidebar.selectbox("Preferred Property Stars", [3, 4, 5])
passport_input  = st.sidebar.selectbox("Has Passport?", ["Yes", "No"])
own_car_input   = st.sidebar.selectbox("Owns a Car?", ["Yes", "No"])
type_contact    = st.sidebar.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
gender          = st.sidebar.selectbox("Gender", ["Male", "Female"])
city_tier       = st.sidebar.selectbox("City Tier", [1, 2, 3])
occupation      = st.sidebar.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
product_pitched = st.sidebar.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
marital_status  = st.sidebar.selectbox("Marital Status", ["Single", "Married", "Divorced"])
designation     = st.sidebar.selectbox("Designation", ["Executive", "Senior Manager", "Manager", "AVP", "VP"])

def preprocess_input(age, monthly_income, num_trips, duration_pitch, num_persons,
                     num_children, num_followups, pitch_score, property_star,
                     passport_input, own_car_input, type_contact, gender, city_tier,
                     occupation, product_pitched, marital_status, designation,
                     expected_features):

    income_per_person = monthly_income / max(num_persons, 1)

    input_data_dict = {
        "Age":                      age,
        "CityTier":                 city_tier,
        "DurationOfPitch":          duration_pitch,
        "NumberOfPersonVisiting":   num_persons,
        "NumberOfFollowups":        num_followups,
        "PreferredPropertyStar":    property_star,
        "NumberOfTrips":            num_trips,
        "PitchSatisfactionScore":   pitch_score,
        "OwnCar":                   1 if own_car_input == "Yes" else 0,
        "NumberOfChildrenVisiting": num_children,
        "MonthlyIncome":            monthly_income,
        "Passport":                 1 if passport_input == "Yes" else 0,
        "income_per_person":        income_per_person,
    }

    input_df = pd.DataFrame([input_data_dict])

    categorical_data = {
        "TypeofContact":  [type_contact],
        "Occupation":     [occupation],
        "Gender":         [gender],
        "ProductPitched": [product_pitched],
        "MaritalStatus":  [marital_status],
        "Designation":    [designation],
    }
    temp_cat_df     = pd.DataFrame(categorical_data)
    encoded_cat_df  = pd.get_dummies(temp_cat_df)
    final_input_df  = pd.concat([input_df, encoded_cat_df], axis=1)
    final_input_df  = final_input_df.reindex(columns=expected_features, fill_value=0)

    return final_input_df

if st.sidebar.button("🔮 Predict Purchase", use_container_width=True):
    processed_input_df = preprocess_input(
        age, monthly_income, num_trips, duration_pitch, num_persons,
        num_children, num_followups, pitch_score, property_star,
        passport_input, own_car_input, type_contact, gender, city_tier,
        occupation, product_pitched, marital_status, designation,
        expected_features
    )

    prediction  = model.predict(processed_input_df)[0]
    probability = model.predict_proba(processed_input_df)[0][1]

    col1, col2, col3 = st.columns(3)
    with col1:
        if prediction == 1:
            st.success("✅ **Likely to Purchase**")
        else:
            st.error("❌ **Unlikely to Purchase**")
    with col2:
        st.metric("Purchase Probability", f"{probability:.1%}")
    with col3:
        st.metric("Model Confidence", "High" if abs(probability - 0.5) > 0.3 else "Medium")

    st.progress(float(probability), text=f"Probability: {probability:.1%}")

    st.subheader("📊 Customer Summary")
    summary = pd.DataFrame({
        "Feature": ["Age", "Monthly Income", "Passport", "Trips/Year", "Occupation"],
        "Value":   [age, f"₹{monthly_income:,}", passport_input, num_trips, occupation]
    })
    st.dataframe(summary, use_container_width=True)
