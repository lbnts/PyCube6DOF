import numpy as np
import time
import matplotlib.pyplot as plt

def plot_vector(x, y, title, xl,yl,l1,l2,l3):
    plt.figure(figsize=(8, 4))
    plt.plot(x, y[:, 0], label=l1)
    plt.plot(x, y[:, 1], label=l2)
    plt.plot(x, y[:, 2], label=l3)
    plt.title(title)
    plt.xlabel(xl)
    plt.ylabel(yl)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_xy(x, y, title, xl,yl,l1):
    plt.figure(figsize=(8, 4))
    plt.plot(x, y, label=l1)
    plt.title(title)
    plt.xlabel(xl)
    plt.ylabel(yl)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def skew(a):
    """Matriz anti-simétrica (produto vetorial matricial)."""
    a1s = np.array([0,-a[2],a[1]])
    a2s = np.array([a[2],0,-a[0]])
    a3s = np.array([-a[1],a[0],0])
    askew = np.array([a1s,a2s,a3s])
    return askew

def dynamics(w,Text,J,dt):
    """Calcula d(omega)/dt de acordo com a Equacao de Newton-Euler."""
    dw = lambda w,Text,J: np.linalg.inv(J) @ (Text -  skew(w) @ (J @ w))

    # RK4 method
    k1 = dw(w,Text,J)
    k2 = dw(w+dt*0.5*k1,Text,J)
    k3 = dw(w+dt*0.5*k2,Text,J)
    k4 = dw(w+dt*k3,Text,J)
    
    # Calcula velocidade angular no prox passo
    dw_rk = (dt/6)*(k1 + 2*k2 + 2*k2 + k4)
    w_next = w + dw_rk
    return w_next

def setup_params():
    "Parametros de simulacao: tempo, incremento"
    dt = 1e-1 # incremento de tempo
    t0 = 0    # tempo inicial
    tf = 100  # tempo final
    n_steps = int(round((tf-t0)/dt,0))  # quantidade de iteracoes
    t_sim = np.linspace(0, tf, n_steps) # array tempo de simulacao
    return dt,t_sim,n_steps

def setup_cubesat():
    "Parametros de massa, inercia, caracteristicas dos sensores e atuadores"
    J = np.diag([1,1,1]) # Matriz de inercia
    return J

def setup_initial_cond():
    "Parametros das condicoes iniciais de simulacao"
    w = np.array([0.1,0.1,0.1]) # Condicao inicial de dinamica
    Text = np.array([0,0,0]) # Torques externos
    return w, Text

def __main__():
    dt, t_sim, n_steps = setup_params()
    J = setup_cubesat()
    w, Text = setup_initial_cond()
    H = J @ w 

    w_data = np.zeros((n_steps,3)) # Prealocar vel angular
    H_data = np.zeros((n_steps,3)) # Prealocar momento angular
    H_norm_data = np.zeros((n_steps,1))
    
    w_data[0,:] = w 
    H_data[0,:] = H
    H_norm_data[0,:] = np.linalg.norm(H_data[0,:])   
    
    for i in range(1,n_steps):
        w = dynamics(w,Text,J,dt)

        # Salvar dados
        w_data[i,:] = w
        H_data[i,:] = J @ w
        H_norm_data[i,:] = np.linalg.norm(H_data[i,:])

    # Plots
    plot_vector(t_sim, w_data, "Dinâmica de Atitude", "Tempo [s]","Vel. angular [rad/s]",'ω₁','ω₂','ω₃')
    plot_vector(t_sim, H_data, "Momento angular", "Tempo [s]","H [kg⋅m²/s]",'H₁','H₂','H₃')
    plot_xy(t_sim, H_norm_data, "Momento angular modulo", "Tempo [s]","H [kg⋅m²/s]",'H')

if __name__ == "__main__":
    __main__()