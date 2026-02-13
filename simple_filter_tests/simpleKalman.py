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


for z in pressure_measurements:
    # predict
    x_pred = x
    P_pred = P + Q

    # and update
    K = P_pred / (P_pred + R)       
    x = x_pred + K * (z - x_pred)   
    P = (1 - K) * P_pred            

    filtered_pressure.append(x)

# basic plotting stuff
plt.figure(figsize=(12, 6))

plt.plot(pressure_measurements, label='Raw Sensor Data', color='lightgray', linestyle='--', marker='o', markersize=3)
plt.plot(filtered_pressure, label='Kalman Filtered', color='blue', linewidth=2)

plt.title('Pressure Sensor: Raw vs. Kalman Filter')
plt.xlabel('Sample Index')
plt.ylabel('Pressure (kPa)')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# plot
plt.tight_layout()
plt.show()