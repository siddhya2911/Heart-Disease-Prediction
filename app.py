# import streamlit as st
# import joblib
# import pandas as pd

# # Load files
# model = joblib.load('KNN_heart.pkl')
# scaler = joblib.load("scaler_heart.pkl")
# expected_columns = joblib.load("columns_heart.pkl")

# st.title("❤️ Heart Disease Prediction by Siddharth")
# st.markdown("Provide the following details:")

# # Inputs
# age = st.slider("Age", 18, 100, 40)
# sex = st.selectbox("Sex", ['M', "F"])
# chest_pain_type = st.selectbox("Chest pain type", ['ATA', 'NAP', 'ASY', 'TA'])
# resting_bp = st.number_input("Resting blood Pressure (mm Hg)", 80, 200, 120)
# cholesterol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
# fasting_bs = st.selectbox("Fasting blood sugar > 120 mg/dl", [0, 1])
# resting_ecg = st.selectbox("Resting ECG", ["Normal", 'ST', 'LVH'])
# max_hr = st.slider("Max Heart Rate", 60, 220, 150)
# exercise_angina = st.selectbox("Exercise-Induced angina", ['Y', 'N'])  # FIXED
# oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
# st_slope = st.selectbox("ST slope", ['Up', 'Flat', 'Down'])

# # Prediction
# if st.button("Prediction"):

#     # Base numeric features
#     input_dict = {
#         'Age': age,
#         'RestingBP': resting_bp,
#         'Cholesterol': cholesterol,
#         'FastingBS': fasting_bs,
#         'MaxHR': max_hr,
#         'Oldpeak': oldpeak,
#     }

#     # Convert to DataFrame
#     input_df = pd.DataFrame([input_dict])

#     # Add ALL expected columns with default 0
#     for col in expected_columns:
#         if col not in input_df.columns:
#             input_df[col] = 0

#     # Set categorical values
#     if 'Sex_M' in expected_columns:
#         input_df['Sex_M'] = 1 if sex == 'M' else 0

#     if f'ChestPainType_{chest_pain_type}' in expected_columns:
#         input_df[f'ChestPainType_{chest_pain_type}'] = 1

#     if f'RestingECG_{resting_ecg}' in expected_columns:
#         input_df[f'RestingECG_{resting_ecg}'] = 1

#     if f'ExerciseAngina_{exercise_angina}' in expected_columns:
#         input_df[f'ExerciseAngina_{exercise_angina}'] = 1

#     if f'ST_Slope_{st_slope}' in expected_columns:
#         input_df[f'ST_Slope_{st_slope}'] = 1

#     # Reorder columns EXACTLY like training
#     input_df = input_df[expected_columns]

#     # Scale input
#     scaled_input = scaler.transform(input_df)

#     # Predict
#     prediction = model.predict(scaled_input)[0]

#     # Output
#     if prediction == 1:
#         st.error("⚠️ High risk of heart disease")
#     else:
#         st.success("✅ Low risk of heart disease")


import streamlit as st
import joblib
import pandas as pd

# --- Page Config ---
st.set_page_config(page_title="Heart Risk Predictor", page_icon="❤️", layout="wide")

# --- Load Model ---
model = joblib.load('KNN_heart.pkl')
scaler = joblib.load("scaler_heart.pkl")
expected_columns = joblib.load("columns_heart.pkl")

# --- Header ---
st.markdown("<h1 style='text-align: center; color: #ff4b4b;'>❤️ Heart Disease Risk Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter patient details to assess heart disease risk</p>", unsafe_allow_html=True)

st.divider()

# --- Layout (2 columns) ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("🧍 Personal Info")
    age = st.slider("Age", 18, 100, 40)
    sex = st.radio("Sex", ['M', 'F'], horizontal=True)

    st.subheader("💓 Heart Metrics")
    resting_bp = st.number_input("Resting BP (mm Hg)", 80, 200, 120)
    cholesterol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
    max_hr = st.slider("Max Heart Rate", 60, 220, 150)

with col2:
    st.subheader("🧪 Medical Details")
    chest_pain_type = st.selectbox("Chest Pain Type", ['ATA', 'NAP', 'ASY', 'TA'])
    fasting_bs = st.radio("Fasting Blood Sugar > 120", [0, 1], horizontal=True)
    resting_ecg = st.selectbox("Resting ECG", ["Normal", 'ST', 'LVH'])

    st.subheader("⚡ Stress Test")
    exercise_angina = st.radio("Exercise-Induced Angina", ['Y', 'N'], horizontal=True)
    oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
    st_slope = st.selectbox("ST Slope", ['Up', 'Flat', 'Down'])

st.divider()

# --- Predict Button ---
predict_btn = st.button("🔍 Predict Risk", use_container_width=True)

if predict_btn:

    # --- Input dictionary ---
    input_dict = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
    }

    input_df = pd.DataFrame([input_dict])

    # Add missing columns
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Encode categorical variables
    input_df['Sex_M'] = 1 if sex == 'M' else 0
    input_df[f'ChestPainType_{chest_pain_type}'] = 1
    input_df[f'RestingECG_{resting_ecg}'] = 1
    input_df[f'ExerciseAngina_{exercise_angina}'] = 1
    input_df[f'ST_Slope_{st_slope}'] = 1

    # Match training column order
    input_df = input_df[expected_columns]

    # Scale
    scaled_input = scaler.transform(input_df)

    # Predict
    prediction = model.predict(scaled_input)[0]

    st.divider()

    # --- Result Display ---
    if prediction == 1:
        st.markdown(
            "<h2 style='text-align:center; color:red;'>⚠️ High Risk of Heart Disease</h2>",
            unsafe_allow_html=True
        )
        st.warning("Consider consulting a cardiologist and improving lifestyle habits.")
    else:
        st.markdown(
            "<h2 style='text-align:center; color:green;'>✅ Low Risk of Heart Disease</h2>",
            unsafe_allow_html=True
        )
        st.success("Keep maintaining a healthy lifestyle!")
