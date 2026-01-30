# Recursive low-pass filter for pressure sensor
'''
    x       : current pressure reading
    y_prev  : previous filtered output
    alpha   : smoothing factor (0 < alpha < 1)
'''

def lowpass_pressure(x, y_prev, alpha=0.1):
    y = alpha * x + (1 - alpha) * y_prev
    return y

pressure_data = [101.3, 101.5, 101.4, 101.7, 101.6, 101.9, 102.0]

alpha = 0.2         
y = pressure_data[0]  

filtered_pressure = [y]

for x in pressure_data[1:]:
    y = lowpass_pressure(x, y, alpha)
    filtered_pressure.append(y)

print("Filtered pressure:", filtered_pressure)
