import numpy as np

def DCM2Euler(R):
    "Converte uma matriz DCM para eixo e ângulo de Euler"
    theta = np.arccos(0.5 * (np.trace(R) - 1))
    ax = (1 / (2 * np.sin(theta))) * (R.T - R)
    a1 = ax[2, 1]
    a2 = ax[0, 2]
    a3 = ax[1, 0]
    a = np.array([a1, a2, a3])
    return a, theta

def Euler2Quat(a, theta):
    "Converte eixo e ângulo de Euler para quaternion (formato [qx, qy, qz, q4])"
    q_vec = a * np.sin(theta / 2)
    q4 = np.cos(theta / 2)
    q = np.concatenate((q_vec, [q4]))
    return q

def Quat2Euler(q):
    "Converte quaternion (Hamilton, [q1 q2 q3 q4]) para matriz DCM"
    qv = q[:3]  # Vetor q1, q2, q3
    q4 = q[3]   # Escalar q4
    R = (q4**2 - np.dot(qv, qv)) * np.eye(3) + 2 * np.outer(qv, qv) - 2 * q4 * skew(qv)
    return R

def skew(a):
    "Retorna a matriz antissimétrica (matriz de produto vetorial)"
    return np.array([
        [0, -a[2], a[1]],
        [a[2], 0, -a[0]],
        [-a[1], a[0], 0]
    ])

# Matriz DCM original
Rab = np.array([
    [0.45457972, 0.43387382, -0.77788868],
    [-0.34766601, 0.89049359, 0.29351236],
    [0.82005221, 0.13702069, 0.55564350]
])

# DCM para eixo-ângulo
a, theta = DCM2Euler(Rab)

# Eixo-ângulo para quaternion
q = Euler2Quat(a, theta)
print("Quaternion [qx, qy, qz, q4]:", q)

# Quaternion para DCM
R = Quat2Euler(q)
print("Matriz DCM reconvertida a partir do quaternion:\n", R)

# Comparação com a original
print("Erro máximo entre Rab e R:", np.max(np.abs(Rab - R)))
