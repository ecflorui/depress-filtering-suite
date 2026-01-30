import numpy as np

# dummy arrary stufff
pressure_measurements = [101.3, 101.5, 101.4, 101.7, 101.6, 101.9, 102.0]

# Kalman filter parameters
Q = 0.0001   # process noise --> i need to tune this for the sensor
R = 0.01     # measurement noise covariance --> i need to tune this for the sensor
x = pressure_measurements[0]  # initial estimate
P = 1.0                       # initial estimate covariance

filtered_pressure = [x]

for z in pressure_measurements[1:]:

    # the prediction step
    x_pred = x
    P_pred = P + Q

    # then we update 
    K = P_pred / (P_pred + R)
    x = x_pred + K * (z - x_pred)
    P = (1 - K) * P_pred

    filtered_pressure.append(x)

print("Filtered pressure:", filtered_pressure)
