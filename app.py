import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import streamlit as st

st.set_page_config(page_title="OpsPulse Telemetry Hub", layout="wide")

# Load and prepare synthetic data
@st.cache_data
def load_data():
    return pd.read_csv("fleet_telemetry.csv")

data = load_data()

# Train instant lightweight Scikit-Learn predictor
X = data[["Device_Age_Months", "Days_Since_Reboot", "Pending_Patches", "BSOD_Last_30D"]]
y = data["Critical_Failure"]
clf = RandomForestClassifier(n_estimators=50, random_state=42)
clf.fit(X, y)

# Sidebar: Persona Switcher and Chaos Mode
st.sidebar.title("OpsPulse Control Center")
view_role = st.sidebar.radio("Active Persona Perspective:", ["CIO / Executive View", "Tier-1 CST Field Tech View"])

st.sidebar.markdown("---")
st.sidebar.subheader("Threat Injection Simulator")
chaos_level = st.sidebar.slider("Ransomware / Zero-Day Outage Pressure (%):", 0, 100, 0, step=10)

# Apply dynamic chaos factor to the fleet
df_view = data.copy()
if chaos_level > 0:
    df_view["Pending_Patches"] += int(chaos_level / 15)
    df_view["Predicted_Failure_Risk"] = np.clip(df_view["Predicted_Failure_Risk"] + (chaos_level * 0.4), 0, 100)

# Top Bar Header
st.title("OpsPulse: Fleet Intelligence & Predictive Telemetry Hub")
st.caption("Bridging Client Systems Operations, Predictive Machine Learning, and Enterprise IT Strategy")

# Metric Cards
col1, col2, col3, col4 = st.columns(4)
fleet_size = len(df_view)
high_risk_count = len(df_view[df_view["Predicted_Failure_Risk"] >= 70])
avg_compliance = 100 - (df_view["Pending_Patches"].sum() / fleet_size * 5)
est_hourly_loss = high_risk_count * 125  # Estimated $125/hr productivity disruption

col1.metric("Monitored Endpoints", f"{fleet_size:,}")
col2.metric("Critical Risk Devices", f"{high_risk_count}", delta=f"{chaos_level}% Threat Level" if chaos_level > 0 else None, delta_color="inverse")
col3.metric("Fleet Patch Compliance", f"{max(0.0, avg_compliance):.1f}%")
col4.metric("Downtime Risk Exposure", f"${est_hourly_loss:,}/hr")

st.markdown("---")

# VIEW 1: CIO / EXECUTIVE VIEW
if view_role == "CIO / Executive View":
    st.subheader("Executive Operations Overview: Strategic Risk & SLAs")
    
    col_chart, col_breakdown = st.columns([2, 1])
    with col_chart:
        st.markdown("**Mean Failure Risk by Department**")
        dept_risk = df_view.groupby("Department")["Predicted_Failure_Risk"].mean().reset_index()
        st.bar_chart(data=dept_risk, x="Department", y="Predicted_Failure_Risk", use_container_width=True)
    
    with col_breakdown:
        st.markdown("**Executive Action Summary**")
        if chaos_level > 50:
            st.error("CRITICAL ADVISORY: Fleet stability compromised. Suspend standard update windows and enact emergency isolation protocols.")
        elif high_risk_count > 150:
            st.warning("ELEVATED RISK: Patch drift is exceeding target SLA thresholds across Finance and Logistics.")
        else:
            st.success("NOMINAL STATE: Fleet SLAs are maintaining 99.2% uptime stability.")

# VIEW 2: TIER-1 FIELD TECH VIEW
else:
    st.subheader("CST Tactical Triage Queue: Prioritized Hardware Remediation")
    st.markdown("Immediate machine-level actions ordered by Scikit-Learn risk score:")
    
    triage_table = df_view[df_view["Predicted_Failure_Risk"] >= 65][
        ["Device_ID", "Department", "Days_Since_Reboot", "Pending_Patches", "BSOD_Last_30D", "Predicted_Failure_Risk"]
    ].sort_values(by="Predicted_Failure_Risk", ascending=False)
    
    st.dataframe(triage_table.head(15), use_container_width=True)
    
    st.download_button(
        label="Export Remediation CSV for Helpdesk Dispatch",
        data=triage_table.to_csv(index=False),
        file_name="triage_dispatch_queue.csv",
        mime="text/csv"
    )

st.markdown("---")

# INTERACTIVE PREDICTION BENCHMARK
st.subheader("Live Device Crash Predictor")
st.markdown("Test single-device crash likelihood using our pre-trained Random Forest classifier:")

p1, p2, p3, p4 = st.columns(4)
test_age = p1.slider("Device Age (Months):", 1, 60, 36)
test_reboot = p2.slider("Days Since Last Reboot:", 0, 60, 14)
test_patches = p3.slider("Missing Security Patches:", 0, 10, 3)
test_bsod = p4.slider("Blue Screen Errors (30 Days):", 0, 5, 2)

if st.button("Run Diagnostic Analysis"):
    pred_risk = clf.predict_proba([[test_age, test_reboot, test_patches, test_bsod]])[0][1] * 100
    if pred_risk >= 60:
        st.error(f"HIGH CRASH PROBABILITY: {pred_risk:.1f}% risk of failure within 7 days. Action: Re-image or replace RAM.")
    else:
        st.success(f"LOW CRASH PROBABILITY: {pred_risk:.1f}% risk of failure within 7 days. Machine is healthy.")
