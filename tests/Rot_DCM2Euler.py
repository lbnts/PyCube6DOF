import numpy as np

def DCM2Euler(R):
    "Obter valor de eixo e ângulo (rad) de Euler a partir da matriz DCM"
    theta = np.arccos(0.5 * (np.trace(R) - 1))  # Ângulo de Euler [rad]
    ax = (1 / (2 * np.sin(theta))) * (R.T - R)
    a1 = ax[2, 1]  # Elemento eixo x
    a2 = ax[0, 2]  # Elemento eixo y
    a3 = ax[1, 0]  # Elemento eixo z
    a = np.array([a1, a2, a3])  # Eixo de Euler
    return a, theta

def Euler2DCM(a, theta):
    # Converte eixo-ângulo de Euler em DCM
    ax = skew(a)
    R = np.cos(theta) * np.eye(3) - np.sin(theta) * ax + (1 - np.cos(theta)) * np.outer(a, a)
    return R

def skew(a):
    "Retorna a matriz antissimétrica (matriz de produto vetorial)"
    skew_a = np.array([
        [0, -a[2], a[1]],
        [a[2], 0, -a[0]],
        [-a[1], a[0], 0]])
    return skew_a

# Matriz DCM
Rab = np.array([[0.45457972, 0.43387382, -0.77788868],
                [-0.34766601, 0.89049359, 0.29351236],
                [0.82005221, 0.13702069, 0.55564350]])

# Obter eixo e ângulo de Euler
a, theta = DCM2Euler(Rab)

# Converter de volta para DCM
R = Euler2DCM(a, theta)

# Impressão opcional dos resultados
print("Eixo a:", a)
print("Ângulo theta (rad):", theta)
print("Matriz R reconvertida:", R)
