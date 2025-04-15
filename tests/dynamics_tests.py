import numpy as np
import time
import matplotlib.pyplot as plt

def skew(a):
    """Matriz anti-simétrica (produto vetorial matricial)."""
    a1s = np.array([0,-a[2],a[1]])
    a2s = np.array([a[2],0,-a[0]])
    a3s = np.array([-a[1],a[0],0])
    askew = np.array([a1s,a2s,a3s])
    return askew

def dynamics(w,Text,J):
    """Calcula d(omega)/dt de acordo com a Equacao de Newton-Euler."""
    dw = lambda w,Text,J: np.linalg.inv(J) @ (Text -  skew(w) @ (J @ w))

    # RK4 method
    k1 = dw(w,Text,J)
    k2 = dw(w+dt*0.5*k1,Text,J)
    k3 = dw(w+dt*0.5*k2,Text,J)
    k4 = dw(w+dt*k3,Text,J)
    
    # Calcula velocidade angular no prox passo
    w_next = w + (dt/6)*(k1 + 2*k2 + 2*k2 + k4)
    return w_next

def setup_params():
    "Parametros de simulacao: tempo, incremento"
    dt = 1e-1 # incremento de tempo
    t0 = 0
    tf = 100
    n_steps = int(round((tf-t0)/dt,0))
    t_sim = np.linspace(0, tf, n_steps)
    return dt,t_sim,n_steps

def setup_cubesat():
    "Parametros de massa, inercia, caracteristicas dos sensores e atuadores"
    J = np.diag([1,2,3]) # Matriz de inercia
    return J

def setup_initial_cond():
    "Parametros das condicoes iniciais de simulacao"
    w = np.array([0,0,0]) # Condicao inicial de dinamica
    Text = np.array([0.01, -0.01, 0.02]) # Torques externos
    return w, Text

dt, t_sim, n_steps = setup_params()
J = setup_cubesat()
w, Text = setup_initial_cond()

w_data = np.zeros((n_steps,3)) # Prealocar
w_data[0,:] = w       

for i in range(1,n_steps):
    w = dynamics(w,Text,J)

    # Salvar dados
    w_data[i,:] = w

# Plot
plt.figure(figsize=(10, 5))
plt.plot(t_sim, w_data[:,0], label='ω₁')
plt.plot(t_sim, w_data[:,1], label='ω₂')
plt.plot(t_sim, w_data[:,2], label='ω₃')

plt.xlabel("Tempo [s]")
plt.ylabel("Velocidade Angular [rad/s]")
plt.title("Dinâmica de Atitude - Newton-Euler (RK4)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
