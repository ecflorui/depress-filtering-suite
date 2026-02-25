import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#dummy csv
data = {
    'time': range(100),
    'pressure': [101.3 + np.random.normal(0, 0.1) for _ in range(100)] # 101.3 baseline + noise
}
df_dummy = pd.DataFrame(data)
df_dummy.to_csv('pressure_data.csv', index=False)
print("Created dummy file: pressure_data.csv")


# loading data
try:
    df = pd.read_csv('pressure_data.csv')
    pressure_measurements = df['pressure'].values
except FileNotFoundError:
    print("Error: 'pressure_data.csv' not found. Please ensure the file exists.")
    exit()

# kalman filter param
Q = 0.0001   # Process noise covariance (Trust in the physics/prediction)
R = 0.01     # Measurement noise covariance (Trust in the sensor)
             # Higher R = Smoother curve but slower reaction to real changes

#intialize
x = pressure_measurements[0]  
P = 1.0                     

filtered_pressure = []
residuals = []   

for z in pressure_measurements:
    # predict
    x_pred = x
    P_pred = P + Q

    # and update
    K = P_pred / (P_pred + R)       
    innovation = z - x_pred         
    x = x_pred + K * innovation   
    P = (1 - K) * P_pred            

    filtered_pressure.append(x)
    residuals.append(innovation)    # NEW

filtered_pressure = np.array(filtered_pressure)
residuals = np.array(residuals)


raw_variance = np.var(pressure_measurements)
filtered_variance = np.var(filtered_pressure)

noise_reduction_percent = (1 - filtered_variance/raw_variance) * 100

# Signal-to-noise ratio estimation
raw_snr = np.mean(pressure_measurements)**2 / raw_variance
filtered_snr = np.mean(filtered_pressure)**2 / filtered_variance
snr_improvement_db = 10 * np.log10(filtered_snr/raw_snr)

# Residual-based anomaly detection (simulating leak detection)
threshold = 3 * np.std(residuals)
anomaly_indices = np.where(np.abs(residuals) > threshold)[0]

print("\n===== PRESSURE SENSOR METRICS =====")
print(f"Raw Variance: {raw_variance:.6f}")
print(f"Filtered Variance: {filtered_variance:.6f}")
print(f"Noise Reduction: {noise_reduction_percent:.2f}%")
print(f"SNR Improvement: {snr_improvement_db:.2f} dB")
print(f"Anomaly Samples (Leak Detection Logic): {anomaly_indices}")
print("====================================\n")


# basic plotting stuff
plt.figure(figsize=(12, 8))

plt.subplot(2,1,1)
plt.plot(pressure_measurements, label='Raw Sensor Data', color='lightgray', linestyle='--', marker='o', markersize=3)
plt.plot(filtered_pressure, label='Kalman Filtered', color='blue', linewidth=2)
plt.title('Pressure Sensor: Raw vs. Kalman Filter')
plt.ylabel('Pressure (kPa)')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)


plt.subplot(2,1,2)
plt.plot(residuals, label='Innovation (Residual Signal)', color='red')
plt.axhline(threshold, linestyle='--', color='black', label='Leak Threshold')
plt.axhline(-threshold, linestyle='--', color='black')
plt.title('Residual Monitoring (Leak / Fault Detection)')
plt.xlabel('Sample Index')
plt.ylabel('Residual')
plt.legend()
plt.grid(True)

# plot
plt.tight_layout()
plt.show()