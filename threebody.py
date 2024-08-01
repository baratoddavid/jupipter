import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Constants
G = 1.0  # Gravitational constant

# Initial conditions
# Positions (x, y) and velocities (vx, vy) for three bodies
m1 = 1.0
m2 = 1.0
m3 = 1.0

# Initial positions and velocities
r1 = np.array([0.0, 1.0])
v1 = np.array([1.0, 0.0])

r2 = np.array([0.0, -1.0])
v2 = np.array([-1.0, 0.0])

r3 = np.array([1.0, 0.0])
v3 = np.array([0.0, 1.0])

# Function to compute the derivatives
def derivatives(t, y):
    r1, r2, r3, v1, v2, v3 = y[:2], y[2:4], y[4:6], y[6:8], y[8:10], y[10:12]
    
    dr1dt = v1
    dr2dt = v2
    dr3dt = v3
    
    r12 = np.linalg.norm(r2 - r1)
    r13 = np.linalg.norm(r3 - r1)
    r23 = np.linalg.norm(r3 - r2)
    
    dv1dt = G * m2 * (r2 - r1) / r12**3 + G * m3 * (r3 - r1) / r13**3
    dv2dt = G * m1 * (r1 - r2) / r12**3 + G * m3 * (r3 - r2) / r23**3
    dv3dt = G * m1 * (r1 - r3) / r13**3 + G * m2 * (r2 - r3) / r23**3
    
    return np.concatenate([dr1dt, dr2dt, dr3dt, dv1dt, dv2dt, dv3dt])

# Initial state vector
y0 = np.concatenate([r1, r2, r3, v1, v2, v3])

# Time span
t_span = (0, 10)
t_eval = np.linspace(*t_span, 1000)

# Solving the ODE
solution = solve_ivp(derivatives, t_span, y0, t_eval=t_eval, rtol=1e-9, atol=1e-9)

# Extracting the solutions
r1_sol = solution.y[:2]
r2_sol = solution.y[2:4]
r3_sol = solution.y[4:6]

# Plotting the results
plt.figure(figsize=(10, 10))
plt.plot(r1_sol[0], r1_sol[1], label='Body 1')
plt.plot(r2_sol[0], r2_sol[1], label='Body 2')
plt.plot(r3_sol[0], r3_sol[1], label='Body 3')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Three-Body Problem')
plt.grid()
plt.show()
