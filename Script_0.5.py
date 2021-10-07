import numpy as np
import matplotlib.pyplot as plt
import time

def extract_weights(f, data):
    (num_bands, _, _, num_vertices) = data.shape

    n = 0
    while n < num_vertices:
        values = f.readline().split()
        for m in range(0, values.__len__()):
            for i in range(0, num_bands):
                for j in range(0, num_bands):
                    data[i][j][0][n] = complex(values[m])
            n += 1
    
    return

def extract_data(f, data):
    (num_bands, _, _, num_vertices) = data.shape

    for n in range(0, num_vertices):
        for k in range(0, num_bands * num_bands):
            values = f.readline().split()
            data[int(values[3]) - 1][int(values[4]) - 1][1][n] = complex(values[0])
            data[int(values[3]) - 1][int(values[4]) - 1][2][n] = complex(values[1])
            data[int(values[3]) - 1][int(values[4]) - 1][3][n] = complex(values[2])
            data[int(values[3]) - 1][int(values[4]) - 1][4][n] = complex(float(values[5]), float(values[6]))

    return

def read_data(file_name):

    with open(file_name, mode='r') as f:
        next(f) #Skip Header
        num_bands = int(f.readline())
        num_vertices = int(f.readline())

        # [Initial_Band][Final_band][Weight:Rx:Ry:Rz:t][Vertex]
        data = np.empty((num_bands, num_bands, 5, num_vertices), dtype=complex)

        extract_weights(f, data)
        extract_data(f, data)

    return data, num_bands, num_vertices

def R_to_K(w, Rx, Ry, Rz, t, Kx, Ky, Kz):
    return (t / w) * np.exp(1j * (Rx * Kx + Ry * Ky + Rz * Kz))

def R_to_K_vectorised(w, Rx, Ry, Rz, t, Kx, Ky, Kz):
    h = np.empty((Kx.__len__()), dtype=complex)

    for n in range(0, Kx.__len__()):
        h[n] = np.sum(R_to_K(w, Rx, Ry, Rz, t, Kx[n], Ky[n], Kz[n]))

    return h

def R_to_K_array(w, Rx, Ry, Rz, t, Kx, Ky, Kz):
    h = np.empty((Kx.__len__(), Ky.__len__(), Kz.__len__()), dtype=complex)

    for i in range(0, Kx.__len__()):
        for j in range(0, Ky.__len__()):
            for k in range(0, Kz.__len__()):
                h[i][j][k] = np.sum(R_to_K(w, Rx, Ry, Rz, t, Kx[i], Ky[j], Kz[i]))

    return h

def diagonalise(m):
    val, vec = np.linalg.eigh(m)
    vec = np.transpose(vec)

    inv_vec = np.linalg.inv(vec)

    diag_m = np.dot(np.dot(inv_vec, m), vec)

    val, vec = np.linalg.eigh(diag_m)

    return diag_m, val, vec


print("Constants:")
a = 3.905*10**-10
z = 0
print("a: " + str(a))
print("z: " + str(z))

print("Importing Data...")
start = time.time()

data, num_bands, num_vertices = read_data("SrTiO3_hr.dat")
print("Imported in " + str(time.time() - start))

print("Route Band Structure and convert it from R space to K space")
start = time.time()
sec_num_points = 100

# RG = R point to Gamma point
Kx_RG = np.linspace(0.5, 0, sec_num_points)
Ky_RG = np.linspace(0.5, 0, sec_num_points)
Kz_RG = np.linspace(0.5, 0, sec_num_points)

# GX = Gamma point to X point
Kx_GX = np.linspace(0, 0.5, sec_num_points)
Ky_GX = np.full(sec_num_points, 0)
Kz_GX = np.full(sec_num_points, 0)

# XM = X point to M point
Kx_XM = np.full(sec_num_points, 0.5 )
Ky_XM = np.linspace(0, 0.5, sec_num_points)
Kz_XM = np.full(sec_num_points, 0)

# MG = M point to Gamma point
Kx_MG = np.linspace(0.5, 0, sec_num_points)
Ky_MG = np.linspace(0.5, 0, sec_num_points)
Kz_MG = np.full(sec_num_points, 0)

# Concatenate Route
Kx = np.concatenate((Kx_RG, Kx_GX[1:], Kx_XM[1:], Kx_MG[1:]))
Ky = np.concatenate((Ky_RG, Ky_GX[1:], Ky_XM[1:], Ky_MG[1:]))
Kz = np.concatenate((Kz_RG, Kz_GX[1:], Kz_XM[1:], Kz_MG[1:]))
num_points = Kx.__len__()

h = np.empty((num_bands, num_bands, num_points), dtype=complex)

for i in range(0, num_bands):
    for j in range(0, num_bands):
        h[i][j] = R_to_K_vectorised(data[i][j][0], data[i][j][1], data[i][j][2], data[i][j][3], data[i][j][4], Kx, Ky, Kz)

print("Converted in " + str(time.time() - start))

print("Diagonalising Band Structure")
start = time.time()

h = np.moveaxis(h, -1, 0)

val = np.empty((num_points, num_bands), dtype=complex)
vec = np.empty((num_points, num_bands, 3), dtype=complex)
for n in range(0, num_points):
    h[n], val[n], vec[n] = diagonalise(h[n])

h = np.moveaxis(h, 0, -1)
val = np.moveaxis(val, 0, -1)
vec = np.moveaxis(vec, 0, -1)

print("Diagonalised in " + str(time.time() - start))


print("Determened bands")

print("Plotting Band Structure:")

for n in range(0, num_bands):
    plt.plot(np.abs(val[n]), label="Band: " + str(n)) 

plt.ylabel("E [eV]")
plot_transitions = (sec_num_points - 1) * np.array(range(0, 5))
plt.xticks(plot_transitions, ['R', 'Γ', 'X', 'M', 'Γ'])
for plot_transition in plot_transitions:
    plt.axvline(x=plot_transition, color='k', linewidth=0.5)
plt.legend(["Band: dzy", "Band: dxz", "Band: dxy"])
plt.show()