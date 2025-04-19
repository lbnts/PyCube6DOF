import numpy as np

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

# Eixos principais
eigx = np.array([1, 0, 0])
eigy = np.array([0, 1, 0])
eigz = np.array([0, 0, 1])

# Vetor a ser rotacionado (no ref A)
vA = np.array([1, 2, 3])

# Ângulos de rotacao
angz =  np.deg2rad(20)  # graus
angy =  np.deg2rad(15)
angx =  np.deg2rad(10)

# Quaternions de rotacao
qz = quat(angz, eigz)
qy = quat(angy, eigy)
qx = quat(angx, eigx)
print(f"qz = {qz}")
print(f"qy = {qy}")
print(f"qx = {qx}")
# Rotacao combinada Q321 = qx ° qy ° qz (de Z para Y para X)
qtemp = qmult(qz, qy)
Q321 = qnorm(qmult(qtemp, qx))  # Normaliza no final
print(f"Q321 = {Q321}")
# Rotacionar vA usando DCM
C321 = quat2dcm(Q321)
print(f"C321 = {C321}")
vB_dir = C321 @ vA

# Rotacionar vA usando quaternion: vB = q ° vA ° q*
qvA = np.array([vA[0], vA[1], vA[2], 0]) # Quaternion do vetor A
q_part = qmult(qconj(Q321), qvA) # resolve vA ° q* = q_part
qvB = qmult(q_part, Q321) # resolve q ° q_part
vB_quat = qvB[:3]

# Comparar com rotacao por quaternions parciais (DCMs separados)
C1 = quat2dcm(qx)
C2 = quat2dcm(qy)
C3 = quat2dcm(qz)
C_parts = C1 @ C2 @ C3
vB_part = C_parts @ vA

print("vB por DCM (Q321):", vB_dir)
print("vB por Quaternions:", vB_quat)
print("vB por DCMs parciais:", vB_part)
print("Erro entre metodos:", np.max(np.abs(vB_quat - vB_dir)))
