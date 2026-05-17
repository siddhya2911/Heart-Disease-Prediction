


# import streamlit as st
# import joblib
# import pandas as pd

# # --- Page Config ---
# st.set_page_config(page_title="Heart Risk Predictor", page_icon="❤️", layout="wide")

# # --- Load Model ---
# model = joblib.load('KNN_heart.pkl')
# scaler = joblib.load("scaler_heart.pkl")
# expected_columns = joblib.load("columns_heart.pkl")

# # --- Header ---
# st.markdown("<h1 style='text-align: center; color: #ff4b4b;'>❤️ Heart Disease Risk Predictor</h1>", unsafe_allow_html=True)
# st.markdown("<p style='text-align: center;'>Enter patient details to assess heart disease risk</p>", unsafe_allow_html=True)

# st.divider()

# # --- Layout (2 columns) ---
# col1, col2 = st.columns(2)

# with col1:
#     st.subheader("🧍 Personal Info")
#     age = st.slider("Age", 18, 100, 40)
#     sex = st.radio("Sex", ['M', 'F'], horizontal=True)

#     st.subheader("💓 Heart Metrics")
#     resting_bp = st.number_input("Resting BP (mm Hg)", 80, 200, 120)
#     cholesterol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
#     max_hr = st.slider("Max Heart Rate", 60, 220, 150)

# with col2:
#     st.subheader("🧪 Medical Details")
#     chest_pain_type = st.selectbox("Chest Pain Type", ['ATA', 'NAP', 'ASY', 'TA'])
#     fasting_bs = st.radio("Fasting Blood Sugar > 120", [0, 1], horizontal=True)
#     resting_ecg = st.selectbox("Resting ECG", ["Normal", 'ST', 'LVH'])

#     st.subheader("⚡ Stress Test")
#     exercise_angina = st.radio("Exercise-Induced Angina", ['Y', 'N'], horizontal=True)
#     oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
#     st_slope = st.selectbox("ST Slope", ['Up', 'Flat', 'Down'])

# st.divider()

# # --- Predict Button ---
# predict_btn = st.button("🔍 Predict Risk", use_container_width=True)

# if predict_btn:

#     # --- Input dictionary ---
#     input_dict = {
#         'Age': age,
#         'RestingBP': resting_bp,
#         'Cholesterol': cholesterol,
#         'FastingBS': fasting_bs,
#         'MaxHR': max_hr,
#         'Oldpeak': oldpeak,
#     }

#     input_df = pd.DataFrame([input_dict])

#     # Add missing columns
#     for col in expected_columns:
#         if col not in input_df.columns:
#             input_df[col] = 0

#     # Encode categorical variables
#     input_df['Sex_M'] = 1 if sex == 'M' else 0
#     input_df[f'ChestPainType_{chest_pain_type}'] = 1
#     input_df[f'RestingECG_{resting_ecg}'] = 1
#     input_df[f'ExerciseAngina_{exercise_angina}'] = 1
#     input_df[f'ST_Slope_{st_slope}'] = 1

#     # Match training column order
#     input_df = input_df[expected_columns]

#     # Scale
#     scaled_input = scaler.transform(input_df)

#     # Predict
#     prediction = model.predict(scaled_input)[0]

#     st.divider()

#     # --- Result Display ---
#     if prediction == 1:
#         st.markdown(
#             "<h2 style='text-align:center; color:red;'>⚠️ High Risk of Heart Disease</h2>",
#             unsafe_allow_html=True
#         )
#         st.warning("Consider consulting a cardiologist and improving lifestyle habits.")
#     else:
#         st.markdown(
#             "<h2 style='text-align:center; color:green;'>✅ Low Risk of Heart Disease</h2>",
#             unsafe_allow_html=True
#         )
#         st.success("Keep maintaining a healthy lifestyle!")



import streamlit as st
import joblib
import pandas as pd
import numpy as np
import time
from streamlit_option_menu import option_menu

# --- Page Config ---
st.set_page_config(
    page_title="Heart Risk Predictor", 
    page_icon="❤️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Adaptive CSS for both light and dark mode ---
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Header styling - Adaptive */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    
    .main-header h1, .main-header p {
        color: white !important;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin: 0.5rem;
        transition: transform 0.3s;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        font-size: 1.1rem;
        font-weight: bold;
        border: none;
        padding: 0.75rem;
        border-radius: 10px;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        color: white !important;
    }
    
    /* Risk level indicators */
    .high-risk {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        animation: pulse 2s infinite;
    }
    
    .high-risk h2, .high-risk p {
        color: white !important;
        margin: 0;
    }
    
    .low-risk {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
    }
    
    .low-risk h2, .low-risk p {
        color: white !important;
        margin: 0;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    
    /* Info box styling - Adaptive for both themes */
    .info-box {
        background-color: var(--secondary-background-color);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
    
    .info-box h4, .info-box p, .info-box li {
        color: var(--text-color) !important;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        border-radius: 5px;
        font-weight: 500;
    }
    
    /* Adaptive text colors using CSS variables */
    .stMarkdown p, .stMarkdown li, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: var(--text-color) !important;
    }
    
    /* Alert message text colors */
    .stAlert p, .stAlert div {
        color: var(--text-color) !important;
    }
    
    /* Form labels */
    .stNumberInput label, .stSelectbox label, .stSlider label, .stRadio label {
        color: var(--text-color) !important;
        font-weight: 500 !important;
    }
    
    /* Radio button text */
    .row-widget.stRadio div {
        color: var(--text-color) !important;
    }
    
    /* Metric values */
    [data-testid="stMetricValue"] {
        color: var(--text-color) !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: var(--text-color) !important;
    }
    
    /* Sidebar text */
    .css-1d391kg .stMarkdown p, .css-1d391kg p {
        color: var(--text-color) !important;
    }
    
    /* Success/Warning/Error/Info message text */
    .stAlert div[data-testid="stMarkdownContainer"] p {
        color: var(--text-color) !important;
    }
    
    /* Select box options */
    div[data-baseweb="select"] div {
        color: var(--text-color) !important;
    }
    
    /* Number input value */
    input[type="number"] {
        color: var(--text-color) !important;
    }
    
    /* Slider value */
    .stSlider div[data-baseweb="slider"] div {
        color: var(--text-color) !important;
    }
    
    /* Divider */
    hr {
        margin: 1rem 0;
        border: 1px solid var(--border-color);
    }
    </style>
""", unsafe_allow_html=True)

# --- Load Model with caching ---
@st.cache_resource
def load_models():
    try:
        model = joblib.load('KNN_heart.pkl')
        scaler = joblib.load("scaler_heart.pkl")
        expected_columns = joblib.load("columns_heart.pkl")
        return model, scaler, expected_columns
    except FileNotFoundError:
        st.error("⚠️ Model files not found! Please ensure all model files are in the correct directory.")
        st.stop()

model, scaler, expected_columns = load_models()

# --- Theme Toggle in Sidebar ---
with st.sidebar:
    st.markdown("## ❤️ Heart Health")
    
    # Theme selector
    theme = st.selectbox(
        "🎨 Theme",
        ["🌞 Light Mode", "🌙 Dark Mode", "🔄 System Default"],
        help="Choose your preferred theme"
    )
    
    if theme == "🌞 Light Mode":
        st._config.set_option("theme.base", "light")
    elif theme == "🌙 Dark Mode":
        st._config.set_option("theme.base", "dark")
    else:
        st._config.set_option("theme.base", "auto")
    
    st.markdown("---")
    
    selected = option_menu(
        menu_title="Navigation",
        options=["Predictor", "Health Tips", "About"],
        icons=["heart-pulse", "lightbulb", "info-circle"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "#ff4b4b", "font-size": "20px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "color": "var(--text-color)"},
            "nav-link-selected": {"background-color": "#ff4b4b", "color": "white !important"},
        }
    )
    
    st.markdown("---")
    st.markdown("### 📊 Quick Stats")
    
    # Stats that adapt to theme
    stats_col1, stats_col2 = st.columns(2)
    with stats_col1:
        st.metric("Global Rate", "31%", "of deaths")
    with stats_col2:
        st.metric("Preventable", "80%", "with lifestyle")

# --- Main Content ---
if selected == "Predictor":
    # Header
    st.markdown("""
        <div class="main-header">
            <h1>❤️ Heart Disease Risk Predictor</h1>
            <p>Advanced AI-powered prediction for early detection</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📝 Patient Information", "📈 Risk Factors", "ℹ️ Guidelines"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**👤 Demographics**")
            age = st.slider(
                "Age (years)", 
                18, 100, 40,
                help="Age is a significant risk factor for heart disease"
            )
            
            sex = st.radio(
                "Sex", 
                ['Male', 'Female'], 
                horizontal=True,
                help="Men generally have higher risk at younger ages"
            )
            
            sex_code = 'M' if sex == 'Male' else 'F'
            
            st.markdown("**💓 Vital Signs**")
            resting_bp = st.number_input(
                "Resting Blood Pressure (mm Hg)", 
                80, 200, 120,
                help="Normal range: 90-120 mm Hg"
            )
            
            max_hr = st.slider(
                "Maximum Heart Rate", 
                60, 220, 150,
                help="Lower max heart rate may indicate higher risk"
            )
        
        with col2:
            st.markdown("**🧪 Laboratory Results**")
            cholesterol = st.number_input(
                "Cholesterol Level (mg/dl)", 
                100, 600, 200,
                help="Desirable: <200 mg/dl"
            )
            
            fasting_bs = st.radio(
                "Fasting Blood Sugar > 120 mg/dl", 
                ['No', 'Yes'], 
                horizontal=True,
                help="High blood sugar is a diabetes indicator"
            )
            
            fasting_bs_code = 1 if fasting_bs == 'Yes' else 0
            
            st.markdown("**📊 Stress Test Results**")
            oldpeak = st.slider(
                "ST Depression (Oldpeak)", 
                0.0, 6.0, 1.0, 0.1,
                help="Higher values indicate greater risk"
            )
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🫀 Chest Pain Analysis**")
            chest_pain_type = st.selectbox(
                "Chest Pain Type",
                ['ATA', 'NAP', 'ASY', 'TA'],
                help="ASY (Asymptomatic) is most concerning"
            )
            
            pain_info = {
                'ATA': 'Typical Angina - Classic chest pain',
                'NAP': 'Non-anginal Pain - Not heart-related typically',
                'ASY': 'Asymptomatic - No symptoms, highest risk',
                'TA': 'Atypical Angina - Vague symptoms'
            }
            st.info(f"ℹ️ {pain_info[chest_pain_type]}")
            
            st.markdown("**🏃 Exercise Response**")
            exercise_angina = st.radio(
                "Exercise-Induced Angina",
                ['No', 'Yes'],
                horizontal=True,
                help="Chest pain during exercise indicates poor blood flow"
            )
            
            exercise_angina_code = 'N' if exercise_angina == 'No' else 'Y'
        
        with col2:
            st.markdown("**📈 ECG Analysis**")
            resting_ecg = st.selectbox(
                "Resting ECG Results",
                ["Normal", 'ST', 'LVH'],
                help="ST abnormalities indicate potential heart issues"
            )
            
            st.markdown("**📐 ST Slope Analysis**")
            st_slope = st.selectbox(
                "ST Slope",
                ['Up', 'Flat', 'Down'],
                help="Down slope indicates highest risk"
            )
            
            slope_info = {
                'Up': 'Upsloping - Favorable',
                'Flat': 'Flat - Moderate risk',
                'Down': 'Downsloping - High risk'
            }
            st.info(f"ℹ️ {slope_info[st_slope]}")
    
    with tab3:
        st.markdown("""
            <div class="info-box">
                <h4>📋 How to use this predictor:</h4>
                <ol>
                    <li>Enter accurate patient information in the "Patient Information" tab</li>
                    <li>Review additional risk factors in the "Risk Factors" tab</li>
                    <li>Click the "🔍 Predict Risk" button below to get results</li>
                    <li>Consult with a healthcare provider for proper medical advice</li>
                </ol>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class="info-box">
                <h4>⚠️ Important Notes:</h4>
                <ul>
                    <li>This tool is for educational purposes only</li>
                    <li>Results should not replace professional medical advice</li>
                    <li>Regular check-ups are essential for heart health</li>
                    <li>Lifestyle changes can significantly reduce risk</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Predict Button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_btn = st.button("🔍 Analyze Heart Risk", use_container_width=True, type="primary")
    
    if predict_btn:
        with st.spinner("🩺 Analyzing patient data..."):
            time.sleep(1)
            
            # Prepare input data
            input_dict = {
                'Age': age,
                'RestingBP': resting_bp,
                'Cholesterol': cholesterol,
                'FastingBS': fasting_bs_code,
                'MaxHR': max_hr,
                'Oldpeak': oldpeak,
            }
            
            input_df = pd.DataFrame([input_dict])
            
            for col in expected_columns:
                if col not in input_df.columns:
                    input_df[col] = 0
            
            input_df['Sex_M'] = 1 if sex_code == 'M' else 0
            input_df[f'ChestPainType_{chest_pain_type}'] = 1
            input_df[f'RestingECG_{resting_ecg}'] = 1
            input_df[f'ExerciseAngina_{exercise_angina_code}'] = 1
            input_df[f'ST_Slope_{st_slope}'] = 1
            
            input_df = input_df[expected_columns]
            
            scaled_input = scaler.transform(input_df)
            prediction = model.predict(scaled_input)[0]
            prediction_prob = model.predict_proba(scaled_input)[0]
        
        st.markdown("---")
        
        # Display results
        if prediction == 1:
            st.markdown("""
                <div class="high-risk">
                    <h2>⚠️ HIGH RISK DETECTED</h2>
                    <p>Probability: {:.1%}</p>
                </div>
            """.format(prediction_prob[1]), unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.error("### 🚨 Immediate Recommendations:")
                st.markdown("""
                    - **Consult a cardiologist urgently**
                    - Consider stress testing
                    - Monitor blood pressure daily
                    - Start heart-healthy diet immediately
                """)
            with col2:
                st.warning("### 📊 Risk Factors Present:")
                risk_factors = []
                if age > 55:
                    risk_factors.append(f"• Age > 55 ({age} years)")
                if resting_bp > 140:
                    risk_factors.append(f"• High blood pressure ({resting_bp} mm Hg)")
                if cholesterol > 240:
                    risk_factors.append(f"• High cholesterol ({cholesterol} mg/dl)")
                if oldpeak > 2:
                    risk_factors.append(f"• Elevated ST depression ({oldpeak})")
                if chest_pain_type == 'ASY':
                    risk_factors.append("• Asymptomatic chest pain type")
                
                if risk_factors:
                    for factor in risk_factors:
                        st.write(factor)
                else:
                    st.write("• Multiple factors combined")
        else:
            st.markdown("""
                <div class="low-risk">
                    <h2>✅ LOW RISK DETECTED</h2>
                    <p>Probability: {:.1%}</p>
                </div>
            """.format(prediction_prob[0]), unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.success("### 🎉 Congratulations!")
                st.markdown("""
                    Your risk assessment shows favorable results. 
                    Maintain your healthy lifestyle habits!
                """)
            with col2:
                st.info("### 💪 Prevention Tips:")
                st.markdown("""
                    - Continue regular exercise
                    - Maintain balanced diet
                    - Schedule annual check-ups
                    - Monitor key health metrics
                """)
        
        # Confidence meter
        st.markdown("### 📊 Prediction Confidence")
        confidence = max(prediction_prob) * 100
        st.progress(int(confidence))
        st.caption(f"Model confidence: {confidence:.1f}%")

elif selected == "Health Tips":
    st.markdown("""
        <div class="main-header">
            <h1>💪 Heart Health Tips</h1>
            <p>Lifestyle changes that can reduce your heart disease risk</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🥗 Nutrition")
        st.info("""
            **Heart-Healthy Diet:**
            - Eat more fruits and vegetables
            - Choose whole grains
            - Include healthy fats (olive oil, avocados)
            - Limit saturated fats and trans fats
            - Reduce sodium intake
            - Eat fatty fish twice weekly
        """)
        
        st.markdown("### 🏃 Physical Activity")
        st.info("""
            **Recommended Exercise:**
            - 150 minutes moderate activity weekly
            - or 75 minutes vigorous activity
            - Include strength training 2x/week
            - Take regular walking breaks
            - Try activities you enjoy
        """)
    
    with col2:
        st.markdown("### 🧘 Stress Management")
        st.info("""
            **Stress Reduction Techniques:**
            - Practice meditation or deep breathing
            - Get adequate sleep (7-9 hours)
            - Maintain social connections
            - Take regular breaks
            - Consider yoga or tai chi
        """)
        
        st.markdown("### 🚫 Risk Factors to Avoid")
        st.info("""
            **Avoid These:**
            - Smoking and tobacco products
            - Excessive alcohol consumption
            - Sedentary lifestyle
            - Chronic stress
            - Processed foods
            - Added sugars
        """)

else:  # About page
    st.markdown("""
        <div class="main-header">
            <h1>ℹ️ About This Tool</h1>
            <p>Understanding heart disease prediction</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
            ### 🤖 How It Works
            
            This predictor uses a **K-Nearest Neighbors (KNN)** machine learning model trained on 
            heart disease patient data. The model analyzes multiple risk factors to assess 
            the likelihood of heart disease.
            
            ### 📊 Key Features Analyzed:
            
            - **Demographics:** Age, Sex
            - **Vital Signs:** Blood pressure, Heart rate
            - **Blood Work:** Cholesterol, Blood sugar
            - **Symptoms:** Chest pain, Exercise-induced angina
            - **ECG Results:** Resting ECG, ST slope
            
            ### 🎯 Model Performance
            
            The model has been validated on clinical data and provides reliable risk assessment 
            when used with accurate input parameters.
            
            ### ⚠️ Disclaimer
            
            This tool is for **educational purposes only** and should not replace professional 
            medical advice. Always consult with healthcare providers for medical decisions.
        """)
    
    with col2:
        st.markdown("### 📞 Resources")
        st.markdown("""
            - [American Heart Association](https://www.heart.org)
            - [WHO Heart Disease Info](https://www.who.int/health-topics/cardiovascular-diseases)
            - [CDC Heart Disease](https://www.cdc.gov/heartdisease)
            - [Find a Cardiologist](https://www.acc.org)
        """)
        
        st.markdown("### 📱 Emergency Signs")
        st.error("""
            **Call emergency services if you experience:**
            - Chest pain/discomfort
            - Shortness of breath
            - Pain in arms/back/neck/jaw
            - Cold sweat
            - Nausea or lightheadedness
        """)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: var(--text-color);'>❤️ Made with Streamlit | AI-Powered Health Prediction | Always consult a doctor</p>",
    unsafe_allow_html=True
)
