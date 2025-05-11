import numpy as np

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



# Ângulos em graus
angz = np.deg2rad(2)
angy = np.deg2rad(1)
angx = np.deg2rad(-3)
rot_type = None  # Escolha do tipo de rotação

R = REuler(angx,angy,angz)

if R is not None:
    print(f"Matriz de rotação para o tipo {rot_type}:\n", R)
else:
    print("Rotação não realizável")
