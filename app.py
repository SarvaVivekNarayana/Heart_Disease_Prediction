import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="wide",
)


@st.cache_resource
def load_artifacts():
    model = joblib.load("knn_heart_model.pkl")
    scaler = joblib.load("heart_scaler.pkl")
    expected_columns = joblib.load("heart_columns.pkl")
    return model, scaler, expected_columns


model, scaler, expected_columns = load_artifacts()

# ---------- Styling ----------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

    html, body, [class*="css"], .stMarkdown, .stButton button {
        font-family: 'Poppins', sans-serif;
    }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1200px; }

    .hero {
        background: linear-gradient(135deg, #e63946 0%, #b5179e 55%, #7209b7 100%);
        border-radius: 22px;
        padding: 2.2rem 2.5rem;
        color: #fff;
        box-shadow: 0 12px 30px rgba(181, 23, 158, 0.28);
        margin-bottom: 1.8rem;
        position: relative;
        overflow: hidden;
    }
    .hero h1 { color: #fff; font-size: 2.3rem; font-weight: 700; margin: 0; }
    .hero p  { color: rgba(255,255,255,0.9); font-size: 1.05rem; margin: 0.4rem 0 0; }
    .hero .pulse {
        position: absolute; right: 2.5rem; top: 50%; transform: translateY(-50%);
        font-size: 4.5rem; animation: beat 1.2s infinite;
    }
    @keyframes beat {
        0%, 100% { transform: translateY(-50%) scale(1); }
        15%      { transform: translateY(-50%) scale(1.18); }
        30%      { transform: translateY(-50%) scale(1); }
        45%      { transform: translateY(-50%) scale(1.12); }
    }

    .section-title {
        font-weight: 600; font-size: 1.1rem; margin: 0 0 0.6rem;
        display: flex; align-items: center; gap: 0.5rem;
    }

    div[data-testid="stForm"] {
        border: 1px solid rgba(128,128,128,0.2);
        border-radius: 20px;
        padding: 1.6rem 1.8rem;
        box-shadow: 0 6px 22px rgba(0,0,0,0.06);
    }

    div[data-testid="stFormSubmitButton"] button {
        width: 100%;
        background: linear-gradient(90deg, #e63946, #b5179e);
        color: #fff; border: none; border-radius: 14px;
        padding: 0.75rem 1rem; font-size: 1.1rem; font-weight: 600;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    div[data-testid="stFormSubmitButton"] button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(230,57,70,0.35);
        color: #fff;
    }

    .result-card {
        border-radius: 20px; padding: 1.8rem 2rem; color: #fff;
        margin-top: 1.5rem; animation: fadeUp 0.5s ease;
    }
    .result-card h2 { color: #fff; margin: 0 0 0.3rem; font-size: 1.8rem; }
    .result-card p  { margin: 0; font-size: 1rem; color: rgba(255,255,255,0.92); }
    .result-high { background: linear-gradient(135deg, #e63946, #9d0208); box-shadow: 0 10px 28px rgba(230,57,70,0.35); }
    .result-low  { background: linear-gradient(135deg, #2a9d8f, #1b7a43); box-shadow: 0 10px 28px rgba(42,157,143,0.35); }
    @keyframes fadeUp { from { opacity: 0; transform: translateY(12px); } to { opacity: 1; transform: none; } }

    .meter { height: 14px; border-radius: 999px; background: rgba(255,255,255,0.25); margin-top: 1rem; overflow: hidden; }
    .meter > span { display: block; height: 100%; border-radius: 999px; background: #fff; }

    .footer { text-align: center; opacity: 0.6; font-size: 0.85rem; margin-top: 2.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Header ----------
st.markdown(
    """
    <div class="hero">
        <h1>Heart Disease Risk Predictor</h1>
        <p>Enter the patient's clinical details below to get an instant risk assessment.</p>
        <p style="opacity:0.8;font-size:0.9rem;margin-top:0.8rem;">Built by Vivek</p>
        <div class="pulse">❤️</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- Input form ----------
CHEST_PAIN = {"ATA": "Atypical Angina (ATA)", "NAP": "Non-Anginal Pain (NAP)",
              "TA": "Typical Angina (TA)", "ASY": "Asymptomatic (ASY)"}
ECG = {"Normal": "Normal", "ST": "ST-T Abnormality (ST)", "LVH": "Left Ventricular Hypertrophy (LVH)"}
SLOPE = {"Up": "Upsloping", "Flat": "Flat", "Down": "Downsloping"}

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        st.markdown('<div class="section-title">👤 Personal</div>', unsafe_allow_html=True)
        age = st.slider("Age", 18, 100, 40)
        sex = st.radio("Sex", ["M", "F"], horizontal=True,
                       format_func=lambda s: "♂ Male" if s == "M" else "♀ Female")
        chest_pain = st.selectbox("Chest Pain Type", list(CHEST_PAIN), format_func=CHEST_PAIN.get)
        exercise_angina = st.radio("Exercise-Induced Angina", ["Y", "N"], horizontal=True,
                                   format_func=lambda s: "Yes" if s == "Y" else "No")

    with col2:
        st.markdown('<div class="section-title">🩸 Vitals & Labs</div>', unsafe_allow_html=True)
        resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
        cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
        fasting_bs = st.radio("Fasting Blood Sugar > 120 mg/dL", [0, 1], horizontal=True,
                              format_func=lambda v: "Yes" if v == 1 else "No")

    with col3:
        st.markdown('<div class="section-title">📈 ECG & Exercise</div>', unsafe_allow_html=True)
        resting_ecg = st.selectbox("Resting ECG", list(ECG), format_func=ECG.get)
        max_hr = st.slider("Max Heart Rate", 60, 220, 150)
        oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0, step=0.1)
        st_slope = st.selectbox("ST Slope", list(SLOPE), format_func=SLOPE.get)

    submitted = st.form_submit_button("🔍 Predict Risk")

# ---------- Prediction ----------
if submitted:
    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }

    input_df = pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]

    scaled_input = scaler.transform(input_df)
    with st.spinner("Analyzing..."):
        prediction = model.predict(scaled_input)[0]
        risk = model.predict_proba(scaled_input)[0][1] * 100

    if prediction == 1:
        st.markdown(
            f"""
            <div class="result-card result-high">
                <h2>⚠️ High Risk of Heart Disease</h2>
                <p>The model estimates a <b>{risk:.0f}%</b> likelihood of heart disease.
                Please consult a cardiologist for a thorough evaluation.</p>
                <div class="meter"><span style="width:{risk:.0f}%"></span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="result-card result-low">
                <h2>✅ Low Risk of Heart Disease</h2>
                <p>The model estimates a <b>{risk:.0f}%</b> likelihood of heart disease.
                Keep up a healthy lifestyle and regular check-ups!</p>
                <div class="meter"><span style="width:{risk:.0f}%"></span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.balloons()

    st.markdown("#### 📊 Input summary")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Age", f"{age} yrs")
    m2.metric("Resting BP", f"{resting_bp} mm Hg")
    m3.metric("Cholesterol", f"{cholesterol} mg/dL")
    m4.metric("Max HR", f"{max_hr} bpm")

st.markdown('<div class="footer">Made with ❤️ using Streamlit · For educational use only</div>',
            unsafe_allow_html=True)
