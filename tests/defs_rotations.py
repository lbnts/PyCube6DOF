import numpy as np

# DCM
def fR1(ang):
    "Rotação no eixo X (Tipo 1) - ângulo em rad"
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(ang), np.sin(ang)],
        [0, -np.sin(ang), np.cos(ang)]])
    return Rx

def fR2(ang):
    "Rotação no eixo Y (Tipo 2) - ângulo em rad"
    Ry = np.array([
        [np.cos(ang), 0, -np.sin(ang)],
        [0, 1, 0],
        [np.sin(ang), 0, np.cos(ang)]])
    return Ry

def fR3(ang):
    "Rotação no eixo Z (Tipo 3) - ângulo em rad"
    Rz =  np.array([
        [np.cos(ang), np.sin(ang), 0],
        [-np.sin(ang), np.cos(ang), 0],
        [0, 0, 1]])
    return Rz

def REuler(angx,angy,angz,rot_type):
    # Matrizes de rotação individuais
    R3 = fR3(angz)
    R2 = fR2(angy)
    R1 = fR1(angx)

    if rot_type is None:
        rot_type = "321" 

    # Dicionário para os tipos de rotação
    rotation_cases = {
        "123": R3 @ R2 @ R1,
        "132": R2 @ R3 @ R1,
        "213": R3 @ R1 @ R2,
        "231": R1 @ R3 @ R2,
        "312": R2 @ R1 @ R3,
        "321": R1 @ R2 @ R3,
        "121": R1 @ R2 @ R1,
        "131": R1 @ R3 @ R1,
        "212": R2 @ R1 @ R2,
        "232": R2 @ R3 @ R2,
        "313": R3 @ R1 @ R3,
        "323": R3 @ R2 @ R3}
    R = rotation_cases.get(rot_type)
    return R

# QUATERNIONS
def quat(ang, eig):
    "Cria um quaternion (Hamilton) a partir de um ângulo [rad] e um eixo unitário"
    q_vec = eig * np.sin(ang/2)
    q4 = np.cos(ang/2)
    return np.concatenate((q_vec, [q4]))

def qnorm(q):
    "Normaliza um quaternion"
    return q / np.linalg.norm(q)

def qconj(q):
    "Retorna o conjugado de um quaternion"
    qc = np.array([-q[0], -q[1], -q[2], q[3]])
    return qc

def qmult(q1, q2):
    "Multiplica dois quaternions (q2 ° q1)"
    #x1, y1, z1, w1 = q1
    #x2, y2, z2, w2 = q2
    qm = omega(q2) @ q1
    return qm

def omega(q):
    "Coloca o quaternion no formato Omega(q) [matriz]"
    x,y,z,w = q
    qmatrix = np.array([[w,z,-y,x],
                       [-z,w,x,y],
                       [y,-x,w,z],
                       [-x,-y,-z,w]])
    return qmatrix

def quat2dcm(q):
    "Converte um quaternion normalizado para matriz DCM"
    x, y, z, w = q
    R = np.array([
        [1 - 2*(y**2 + z**2),     2*(x*y + z*w),     2*(x*z - y*w)],
        [    2*(x*y - z*w), 1 - 2*(x**2 + z**2),     2*(y*z + x*w)],
        [    2*(x*z + y*w),     2*(y*z - x*w), 1 - 2*(x**2 + y**2)]])
    return R 

# CONVERSOES ENTRE QUATERNIONS, DCM E EULER
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