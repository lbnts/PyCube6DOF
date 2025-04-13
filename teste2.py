import numpy as np
import matplotlib.pyplot as plt

def skew(v):
    """Matriz anti-simétrica (produto vetorial matricial)."""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

def newton_euler_rhs(omega, T_ext, J):
    """Calcula d(omega)/dt segundo Newton-Euler."""
    J_inv = np.linalg.inv(J)
    cross_term = np.cross(omega, J @ omega)
    return J_inv @ (T_ext - cross_term)

def runge_kutta4_step(f, omega, t, dt, T_ext, J):
    """Um passo do método de Runge-Kutta de 4ª ordem."""
    k1 = f(omega, T_ext, J)
    k2 = f(omega + 0.5 * dt * k1, T_ext, J)
    k3 = f(omega + 0.5 * dt * k2, T_ext, J)
    k4 = f(omega + dt * k3, T_ext, J)
    return omega + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def simular_dinamica(J, omega0, T_ext_func, t_final=100.0, dt=0.1):
    """Simula a dinâmica de rotação usando RK4."""
    num_steps = int(t_final / dt) + 1
    t_array = np.linspace(0, t_final, num_steps)
    omega_array = np.zeros((num_steps, 3))

    omega = omega0.copy()
    for i, t in enumerate(t_array):
        omega_array[i] = omega
        T = T_ext_func(t)
        omega = runge_kutta4_step(newton_euler_rhs, omega, t, dt, T, J)
    
    return t_array, omega_array

# Parâmetros
J = np.diag([1.0, 2.0, 3.0])  # Matriz de inércia
omega0 = np.array([0.0, 0.0, 0.0])  # Velocidade angular inicial

# Torque constante ao longo do tempo
def T_ext_func(t):
    return np.array([0.01, -0.01, 0.02])

# Simulação
t_array, omega_array = simular_dinamica(J, omega0, T_ext_func)

# Plot
plt.figure(figsize=(10, 5))
plt.plot(t_array, omega_array[:, 0], label='ω₁')
plt.plot(t_array, omega_array[:, 1], label='ω₂')
plt.plot(t_array, omega_array[:, 2], label='ω₃')
plt.xlabel("Tempo [s]")
plt.ylabel("Velocidade Angular [rad/s]")
plt.title("Dinâmica de Atitude - Newton-Euler (RK4)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
