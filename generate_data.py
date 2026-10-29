import numpy as np
import pandas as pd

np.random.seed(42)
n_devices = 1000

device_ids = [f"USAF-LPT-{1000 + i}" for i in range(n_devices)]
departments = np.random.choice(["Operations", "Logistics", "Cyber/IT", "Finance", "Medical"], size=n_devices)
device_age_months = np.random.randint(6, 49, size=n_devices)
days_since_reboot = np.random.exponential(scale=5, size=n_devices).astype(int)
pending_critical_patches = np.random.poisson(lam=1.5, size=n_devices)
blue_screen_events_30d = np.random.poisson(lam=0.4, size=n_devices)
antivirus_active = np.random.choice([True, False], size=n_devices, p=[0.94, 0.06])

# Calculate failure probability based on telemetry patterns
risk_score = (
    (device_age_months * 0.8) +
    (days_since_reboot * 2.5) +
    (pending_critical_patches * 12.0) +
    (blue_screen_events_30d * 20.0) +
    (~antivirus_active * 35.0)
)
# Normalize to 0-100%
risk_prob = 1 / (1 + np.exp(- (risk_score - 50) / 15))
failed_within_7d = (risk_prob > 0.65).astype(int)

df = pd.DataFrame({
    "Device_ID": device_ids,
    "Department": departments,
    "Device_Age_Months": device_age_months,
    "Days_Since_Reboot": days_since_reboot,
    "Pending_Patches": pending_critical_patches,
    "BSOD_Last_30D": blue_screen_events_30d,
    "Antivirus_Active": antivirus_active,
    "Predicted_Failure_Risk": np.round(risk_prob * 100, 1),
    "Critical_Failure": failed_within_7d
})

df.to_csv("fleet_telemetry.csv", index=False)
print("SUCCESS: fleet_telemetry.csv generated with 1,000 endpoint records.")