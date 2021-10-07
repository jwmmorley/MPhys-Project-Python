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

def hamiltonian(Rx, Ry, Rz, t, Kx, Ky, a, z):
    return (np.sqrt(2 * np.pi) / a) * t * np.exp(1j * ((Kx * Rx) + (Ky * Ry))) * np.sinc((np.pi / a) * (Rz + z))

def hamiltonian_array(Rx, Ry, Rz, t, Kx, Ky, a, z):
    h = np.empty((Kx.__len__(), Ky.__len__()), dtype=complex)

    for i in range(0, Kx.__len__()):
        for j in range(0, Ky.__len__()):
            h[i][j] = np.sum(hamiltonian(Rx, Ry, Rz, t, Kx[i], Ky[j], a, z))

    return h

def band_structure(Rx, Ry, Rz, t, k_sensitivity, a, z):
    k_range = np.pi / a
    k_step = 2 * k_range / k_sensitivity
    Kx = np.concatenate([np.linspace(0, k_range + k_step, k_sensitivity), np.full(k_sensitivity, k_range + k_step, dtype=float), np.linspace(k_range + k_step, 0, k_sensitivity)])
    Ky = np.concatenate([np.full(k_sensitivity, 0, dtype=float), np.linspace(0, k_range + k_step, k_sensitivity), np.linspace(k_range + k_step, 0, k_sensitivity)])

    h = np.empty((Kx.__len__()), dtype=complex)

    for i in range(0, Kx.__len__()):
        h[i] = np.sum(hamiltonian(Rx, Ry, Rz, t, Kx[i], Ky[i], a, z))

    return h


print("Constants:")
a = 3.905*10**-10
z = 0
print("a: " + str(a))
print("z: " + str(z))

print("Importing Data...")
start = time.time()
data, num_bands, num_vertices = read_data("SrTiO3_hr.dat")
print("Imported in " + str(time.time() - start))

print("Building hamiltonian...")
start = time.time()
k_range = np.pi / a
k_sensitivity = 10
k_step = 2 * k_range / k_sensitivity
Kx = np.linspace(-k_range, k_range + k_step, k_sensitivity * 2)
Ky = np.linspace(-k_range, k_range + k_step, k_sensitivity * 2)

h = hamiltonian_array(data[0][0][1], data[0][0][2], data[0][0][3], data[0][0][4], Kx, Ky, a, z)

plt.contourf(Kx, Ky, h)
plt.show()
print("Built in " + str(str(time.time() - start)))

print("Building Band Structure...")
start = time.time()
b = band_structure(data[0][0][1], data[0][0][2], data[0][0][3], data[0][0][4], 10, a, z)

plt.plot(b)
print("Built in " + str(str(time.time() - start)))