import numpy as np
from defs_rotations import *

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

# Rotacao combinada Q321 = qx ° qy ° qz (de Z para Y para X)
qtemp = qmult(qz, qy)
Q321 = qnorm(qmult(qtemp, qx))  # Normaliza no final

# Rotacionar vA usando DCM
C321 = quat2dcm(Q321)
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