import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from dotenv import load_dotenv

# Gemini
import google.generativeai as genai

# --------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------
MODEL_PATH = "models/global_model.pkl"
METRICS_PATH = "models/fl_metrics.pkl"

# --------------------------------------------------------------------
# Environment & Gemini setup
# --------------------------------------------------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        GEMINI_MODEL = genai.GenerativeModel("gemini-2.5-flash")
    except Exception as e:
        print(f"[WARN] Gemini init failed: {e}")
        GEMINI_MODEL = None
else:
    print("[INFO] GEMINI_API_KEY not set; using rule-based explanation.")
    GEMINI_MODEL = None


# --------------------------------------------------------------------
# Model & metrics loading
# --------------------------------------------------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("Global model not found. Run federated_training.py first.")
        return None, None

    try:
        with open(MODEL_PATH, "rb") as f:
            obj = pickle.load(f)

        model = obj.get("model")
        scaler = obj.get("scaler")

        if model is None or scaler is None:
            st.error("Model or scaler missing in pickle file.")
            return None, None

        return model, scaler
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None


@st.cache_resource
def load_fl_metrics():
    if not os.path.exists(METRICS_PATH):
        return None

    try:
        with open(METRICS_PATH, "rb") as f:
            obj = pickle.load(f)
        return obj.get("auc_history", None)
    except Exception:
        return None


# --------------------------------------------------------------------
# Risk breakdown & explanation helpers
# --------------------------------------------------------------------
def compute_component_risks(features, base_prob: float):
    """
    Simple heuristic to show component-level risks (0–1).
    Just for visualization – not a scientific model.
    """
    engine_risk = max(0.0, (features["avg_engine_temp"] - 85) / 40)
    cooling_risk = max(0.0, (features["max_engine_temp"] - 100) / 50)
    suspension_risk = max(0.0, (features["vibration_level"] - 0.7) / 1.3)
    usage_risk = max(0.0, (features["mileage"] - 80_000) / 120_000)

    risks = {
        "Engine Wear": float(np.clip(engine_risk, 0, 1)),
        "Cooling System": float(np.clip(cooling_risk, 0, 1)),
        "Suspension / Chassis": float(np.clip(suspension_risk, 0, 1)),
        "Usage Fatigue": float(np.clip(usage_risk, 0, 1)),
    }

    for k in risks:
        risks[k] = float(np.clip(risks[k] * (0.5 + base_prob), 0, 1))

    return risks


def generate_explanation_rule_based(features, prob):
    reasons = []

    if features["avg_engine_temp"] > 95:
        reasons.append("high average engine temperature over time")
    if features["max_engine_temp"] > 115:
        reasons.append("frequent overheating spikes")
    if features["vibration_level"] > 1.0:
        reasons.append("elevated vibration levels indicating mechanical wear")
    if features["maintenance_gap"] > 20000:
        reasons.append("long gap since last maintenance")
    if features["mileage"] > 120000:
        reasons.append("very high mileage suggesting aging components")
    if features["speed_variance"] > 20:
        reasons.append("irregular driving patterns increasing stress on parts")

    if not reasons:
        reasons.append("no strong anomalies detected in monitored signals")

    text = (
        f"The estimated failure risk for this vehicle is **{prob*100:.1f}%**. "
        f"Key contributing factors: {', '.join(reasons)}. "
        "Based on this risk level, a preventive inspection at an authorized service center "
        "is recommended to avoid unplanned breakdowns."
    )
    return text


def generate_explanation_gemini(features, prob):
    """
    Use Gemini to generate a technician-style diagnosis.
    Falls back to rule-based if Gemini not available.
    """
    if GEMINI_MODEL is None:
        return generate_explanation_rule_based(features, prob)

    prompt = f"""
You are an automotive diagnostic AI assistant. Explain this vehicle's condition
for a service technician in clear, concise language.

Failure Probability: {prob*100:.1f}%

Vehicle Telemetry:
- Mileage (km): {features['mileage']}
- Average Engine Temperature (°C): {features['avg_engine_temp']}
- Maximum Engine Temperature (°C): {features['max_engine_temp']}
- Vibration Level: {features['vibration_level']}
- Speed Variance: {features['speed_variance']}
- Kilometers Since Last Service: {features['maintenance_gap']}

Provide:
1. A 3–4 sentence diagnosis in simple language.
2. The most likely root cause or subsystem(s) at risk.
3. A clear recommended action and urgency (e.g., “within 3 days / next 500 km”).
Keep it concise and professional.
"""

    try:
        response = GEMINI_MODEL.generate_content(prompt)
        if response and response.text:
            return response.text
        else:
            return generate_explanation_rule_based(features, prob)
    except Exception as e:
        # Log in console, but don't break the app
        print(f"[WARN] Gemini explanation failed, falling back. Error: {e}")
        return generate_explanation_rule_based(features, prob)


def simulate_history(features, n_points: int = 10):
    """
    Fake 'last N trips' history based on current input,
    purely for visualization in the demo.
    """
    trips = np.arange(1, n_points + 1)
    avg_temp_hist = np.random.normal(features["avg_engine_temp"], 2.0, n_points)
    vib_hist = np.random.normal(features["vibration_level"], 0.05, n_points)

    df = pd.DataFrame(
        {
            "Trip": trips,
            "Avg Engine Temp (°C)": avg_temp_hist,
            "Vibration Level": vib_hist,
        }
    )
    return df.set_index("Trip")


# --------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="Federated Agentic AI – Predictive Vehicle Maintenance",
        layout="wide",
    )

    st.title("🚗 Federated Agentic AI – Predictive Vehicle Maintenance")
    st.markdown(
        """
This dashboard showcases a **Federated Learning global model** trained across simulated vehicle nodes,
combined with an **Agentic AI explanation layer powered by Gemini** for predictive maintenance.
"""
    )

    model, scaler = load_model()
    if model is None or scaler is None:
        st.stop()

    auc_history = load_fl_metrics()

    # ---------------- Sidebar inputs ----------------
    st.sidebar.header("Vehicle Telemetry Input")

    mileage = st.sidebar.slider("Mileage (km)", 5_000, 200_000, 60_000, step=5_000)
    avg_temp = st.sidebar.slider("Average Engine Temp (°C)", 70, 120, 90)
    max_temp = st.sidebar.slider("Max Engine Temp (°C)", 80, 150, 115)
    vibration = st.sidebar.slider("Vibration Level", 0.1, 2.0, 0.8, step=0.05)
    speed_var = st.sidebar.slider("Speed Variance", 2, 40, 12, step=1)
    maint_gap = st.sidebar.slider(
        "Km Since Last Service", 2_000, 40_000, 10_000, step=2_000
    )

    features_arr = np.array(
        [[mileage, avg_temp, max_temp, vibration, speed_var, maint_gap]]
    )
    features_dict = {
        "mileage": mileage,
        "avg_engine_temp": avg_temp,
        "max_engine_temp": max_temp,
        "vibration_level": vibration,
        "speed_variance": speed_var,
        "maintenance_gap": maint_gap,
    }

    col_main, col_side = st.columns([1.4, 1.6])

    # ---------------- Prediction + Visuals ----------------
    with col_main:
        st.subheader("Prediction Panel")

        if st.button("Predict Failure Risk"):
            try:
                scaled = scaler.transform(features_arr)
            except Exception as e:
                st.error(f"Scaling error: {e}")
                st.stop()

            try:
                prob = model.predict_proba(scaled)[0, 1]
            except Exception:
                score = model.decision_function(scaled)[0]
                prob = 1 / (1 + np.exp(-score))

            pred_label = "🚨 HIGH RISK" if prob > 0.5 else "✅ LOW RISK"

            st.metric("Failure Risk Probability", f"{prob*100:.1f}%")
            st.write(f"Status: **{pred_label}**")

            st.write("Overall Risk Gauge:")
            st.progress(float(np.clip(prob, 0, 1)))

            component_risks = compute_component_risks(features_dict, prob)
            st.subheader("Component-wise Risk (Simulated)")
            fig, ax = plt.subplots()
            ax.bar(component_risks.keys(), component_risks.values())
            ax.set_ylim(0, 1)
            ax.set_ylabel("Relative Risk (0–1)")
            ax.tick_params(axis="x", rotation=20)
            st.pyplot(fig)

            # ---------------- Right column: AI + history ----------------
            with col_side:
                st.subheader("AI Explanation")

                explanation = generate_explanation_gemini(features_dict, prob)
                st.write(explanation)

                st.subheader("Input Snapshot")
                st.json(features_dict)

                st.subheader("Recent Trips – Trend View (Simulated)")
                hist_df = simulate_history(features_dict)
                st.line_chart(hist_df)
        else:
            st.info("Adjust the sliders and click **Predict Failure Risk** to run the model.")

    # ---------------- FL metrics chart ----------------
    st.markdown("---")
    st.subheader("Federated Learning Convergence – Global Model AUC per Round")
    if auc_history is not None and len(auc_history) > 0:
        rounds = list(range(1, len(auc_history) + 1))
        df_auc = pd.DataFrame({"Round": rounds, "Global AUC": auc_history}).set_index(
            "Round"
        )
        st.line_chart(df_auc)
        st.caption(
            "This chart shows how the **global model** improves as simulated vehicle nodes "
            "train locally and share only their model updates (FedAvg)."
        )
    else:
        st.write(
            "Federated metrics not found. Re-run federated_training.py to regenerate AUC history."
        )


if __name__ == "__main__":
    main()
