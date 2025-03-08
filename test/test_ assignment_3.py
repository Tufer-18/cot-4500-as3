import numpy as np

def euler_method(f, t0, y0, t_end, n):
    h = (t_end - t0) / n  # Step size
    t = t0
    y = y0
    
    for _ in range(n):
        y += h * f(t, y)
        t += h
    
    return y

def runge_kutta_method(f, t0, y0, t_end, n):
    h = (t_end - t0) / n  # Step size
    t = t0
    y = y0
    
    for _ in range(n):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h * k1 / 2)
        k3 = f(t + h / 2, y + h * k2 / 2)
        k4 = f(t + h, y + h * k3)
        y += (h / 6) * (k1 + 2*k2 + 2*k3 + k4)
        t += h
    
    return y

# Given function
def function(t, y):
    return t - y**2

# Parameters
t0 = 0
y0 = 1
t_end = 2
n = 10

# Compute results using both methods
euler_result = euler_method(function, t0, y0, t_end, n)
rk4_result = runge_kutta_method(function, t0, y0, t_end, n)

# Print results
print(f"{euler_result}")
print(f"{rk4_result}")


