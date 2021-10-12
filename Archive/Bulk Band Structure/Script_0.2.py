import numpy as np
import matplotlib.pyplot as plt
import time


def extract_weights(f, num_vertices):
    temp = np.empty(num_vertices, dtype=int)

    n = 0 # Vertex
    while n < num_vertices:
        values = f.readline().split()
        for i in range(0, values.__len__()):
            temp[n] = int(values[i])
            n += 1

    return f, temp


def extract_data(f, num_bands, num_vertices):
    temp = np.empty((num_bands, num_bands, num_vertices, 4), dtype=complex)

    for n in range(0, num_vertices):
        for i in range(0, num_bands):
            for j in range(0, num_bands):
                values = f.readline().split()
                temp[i][j][n] = [complex(values[0]), complex(values[1]), complex(values[2]), complex(float(values[5]), float(values[6]))]

    return f, temp


def read_data(file_name):

    with open(file_name, mode='r') as f:
        next(f) #Skip Header
        num_bands = int(f.readline())
        num_vertices = int(f.readline())

        f, weights = extract_weights(f, num_vertices)
        f, data = extract_data(f, num_bands, num_vertices)

    return num_bands, num_vertices, weights, data

"""
def calc_hamiltonian(d, k, a, z): # Use Sinc version!
    num_vertices = d.shape[0]

    s = 0
    for n in range(0, num_vertices):
        if (d[n][2] + z) == 0: # Using l'Hospital!!!
            s += d[n][3] * np.exp(1j * (k[0] * d[n][0] + k[1] * d[n][1])) * np.pi / a # Never triggers!!
        else:
            s += d[n][3] * np.exp(1j * (k[0] * d[n][0] + k[1] * d[n][1])) * np.sin(np.pi * (d[n][2] + z) / a) / (d[n][2] + z)

    return s * 2 / np.sqrt(2)
"""

def hamiltonian(d, k, a, z):
    num_vertices = d.shape[0]

    s = 0
    for n in range(0, num_vertices):
        s += d[n][3] * np.exp(1j * (k[0] * d[n][0] + k[1] * d[n][1])) * np.sinc(np.pi * (d[n][2] + z) / a)

    return s * 2 * np.pi / (np.sqrt(2 * np.pi) * a)

def hamiltonian_test(rx, ry, rz, t, kx, ky, a, z):
    temp = t * np.exp(1j * (kx * rx + ky * ry)) * np.sinc(np.pi * (rz + z) / a) * 2 * np.pi / (np.sqrt(2 * np.pi) * a)
    print(rx)
    return np.sum(temp)
    
def build_hamiltonian(d, k, a, z):
    num_bands = d.shape[0]
    temp = np.empty((num_bands, num_bands), dtype=complex)

    for i in range(0, num_bands):
        for j in range(0, num_bands):
            temp[i][j] = hamiltonian(data[i][j], k, a, z)
            #temp[i][j] = hamiltonian_test(data[i, j][0], data[i, j][1], data[i, j][2], data[i, j][3], k[0], k[1], a, z)

    return temp


print("Constants:")
a = 3.905*10**-10
z = 10
print("a: " + str(a))
print("z: " + str(z))

k_range = np.pi / a
k_sensitivity = 10
k_step = 2 * k_range / k_sensitivity
kx = np.linspace(-k_range, k_range + k_step, k_sensitivity * 2)
ky = np.linspace(-k_range, k_range + k_step, k_sensitivity * 2)
#ks = np.mgrid[-k_range:k_range + k_step:k_step, -k_range:k_range + k_step:k_step].reshape(2,-1).T

print("Importing Data...")
start = time.time()
num_bands, num_vertices, weights, data = read_data("SrTiO3_hr.dat")
print("Imported in " + str(time.time() - start))

print("Building Hamiltonian...")
start = time.time()
h = np.empty((kx.__len__(), ky.__len__()), dtype=complex)
for x, y in np.ndindex(h.shape):
    h[x][y] = build_hamiltonian(data, (kx[x], ky[y]), a, z)[0][0]
print("Built Hamiltonian in " + str(time.time() - start))

plt.contourf(kx, ky, h)
plt.show()