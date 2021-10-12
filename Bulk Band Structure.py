import matplotlib
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
    return np.sum((t / w) * np.exp(-1j * (Rx * Kx + Ry * Ky + Rz * Kz)))

def R_to_K_array(w, Rx, Ry, Rz, t, Kx, Ky, Kz):
    h = np.empty((Kx.__len__(), Ky.__len__(), Kz.__len__()), dtype=complex)

    for i in range(0, Kx.__len__()):
        for j in range(0, Ky.__len__()):
            for k in range(0, Kz.__len__()):
                h[i][j][k] = R_to_K(w, Rx, Ry, Rz, t, Kx[i], Ky[j], Kz[i])

    return h

def K_to_KZ(h, Kz, z):
    return np.sum()

def diagonalise(m):
    val, vec = np.linalg.eigh(m)
    inv_vec = np.linalg.inv(vec)
    diag_m = np.dot(np.dot(inv_vec, m), vec)

    val, vec = np.linalg.eigh(diag_m)
    return diag_m, val, vec

def least_variation(arrays):
    min_dif = np.max(arrays[0])
    index = 0
    for i in range(0, arrays.__len__()):
        dif = np.max(arrays[i]) - np.min(arrays[i])
        if dif < min_dif:
            min_dif = dif
            index = i
    return index



debug_times = np.array([], dtype='f8')
debug_labels = np.array([], dtype='U16')
start_time = time.time()


print("Constants:")
a = 1 # 3.905*10**-10
z = 0
print("a: " + str(a))
print("z: " + str(z))


print("Importing Data...")
start = time.time()
data, num_bands, num_vertices = read_data("SrTiO3_hr.dat")
debug_times = np.append(debug_times, time.time() - start)
debug_labels = np.append(debug_labels, "Import")
print("Imported in " + str(debug_times[-1]))



print("Create Bulk Band Structure route...")
start = time.time()
edge = np.pi
step_length = edge / 1000
sec_num_points_base = int(edge / step_length)

# RG = R point to Gamma point
sec_num_points = int(sec_num_points_base * np.sqrt(3))
Kx_RG = np.linspace(edge, 0, sec_num_points)
Ky_RG = np.linspace(edge, 0, sec_num_points)
Kz_RG = np.linspace(edge, 0, sec_num_points)

# GX = Gamma point to X point
sec_num_points = sec_num_points_base
Kx_GX = np.linspace(0, edge, sec_num_points)
Ky_GX = np.full(sec_num_points, 0)
Kz_GX = np.full(sec_num_points, 0)

# XM = X point to M point
sec_num_points = sec_num_points_base
Kx_XM = np.full(sec_num_points, edge)
Ky_XM = np.linspace(0, edge, sec_num_points)
Kz_XM = np.full(sec_num_points, 0)

# MG = M point to Gamma point
sec_num_points = int(sec_num_points_base * np.sqrt(2))
Kx_MG = np.linspace(edge, 0, sec_num_points)
Ky_MG = np.linspace(edge, 0, sec_num_points)
Kz_MG = np.full(sec_num_points, 0)

# Concatenate Route
Kx = np.concatenate((Kx_RG, Kx_GX[1:], Kx_XM[1:], Kx_MG[1:]))
Ky = np.concatenate((Ky_RG, Ky_GX[1:], Ky_XM[1:], Ky_MG[1:]))
Kz = np.concatenate((Kz_RG, Kz_GX[1:], Kz_XM[1:], Kz_MG[1:]))
num_points = Kx.__len__()
print("Number of points: " + str(num_points))
debug_times = np.append(debug_times, time.time() - start)
debug_labels = np.append(debug_labels, "Route Created")
print("Route created in " + str(debug_times[-1]))


print("Convert from R space to K space...")
start = time.time()
h = np.empty((num_bands, num_bands, num_points), dtype=complex)

for i in range(0, num_bands):
    for j in range(0, num_bands):
        for k in range(0, num_points):
            h[i][j][k] = R_to_K(data[i][j][0], data[i][j][1], data[i][j][2], data[i][j][3], data[i][j][4], Kx[k], Ky[k], Kz[k])

h = np.moveaxis(h, -1, 0) # [Vertex][i][j]
debug_times = np.append(debug_times, time.time() - start)
debug_labels = np.append(debug_labels, "Converted")
print("Converted in " + str(debug_times[-1]))


print("Diagonalising Bulk Band Structure...")
start = time.time()
val = np.empty((num_points, num_bands), dtype=complex)
vec = np.empty((num_points, num_bands, 3), dtype=complex)

for n in range(0, num_points):
    h[n], val[n], vec[n] = diagonalise(h[n])
debug_times = np.append(debug_times, time.time() - start)
debug_labels = np.append(debug_labels, "Diagonalised")
print("Diagonalised in " + str(debug_times[-1]))


print("Subtracting Conducting Band Minimum Energy from Bulk Band Structure...")
start = time.time()
val = np.subtract(val, np.min(val))
debug_times = np.append(debug_times, time.time() - start)
debug_labels = np.append(debug_labels, "Subtraction")
print("Subtracted in " + str(debug_times[-1]))


print("Finding Weights for each band...")
start = time.time()

weights = np.empty((num_points, num_bands, num_bands), dtype=float)
for n in range(0, num_points):
    for i in range(0, num_bands):
        for j in range(0, num_bands):
            weights[n][i][j] = np.abs(vec[n][i][j])**2
weights = (weights - np.min(weights)) / np.max(weights)
debug_times = np.append(debug_times, time.time() - start)
debug_labels = np.append(debug_labels, "Weights")
print("Found Weights in " + str(debug_times[-1]))


print("Plotting Band Structure...")
start = time.time()
dxz_index = least_variation(np.moveaxis(val[(Kx_RG.__len__() + Kx_GX.__len__() - 2):(Kx_RG.__len__() + Kx_GX.__len__() + Kx_XM.__len__() - 3)], 0, -1))
dzy_index = least_variation(np.moveaxis(val[(Kx_RG.__len__() - 1):(Kx_RG.__len__() + Kx_GX.__len__() - 2)], 0, -1))
dxy_index = 3 - (dxz_index + dzy_index)

val = np.moveaxis(val, 0, -1) # [band][Vertex]
weights = np.moveaxis(weights, 1, 0) # [i][Vertex][j]
for n in range(0, num_bands):
    band = str(n)
    if band == dxy_index:
        band = "dxy"
    elif band == dxz_index:
        band = "dxz"
    elif band == dzy_index:
        band = "dzy"
    #plt.plot(np.real(val[n]), label="Band: " + band)
    plt.scatter(range(0, num_points), np.real(val[n]), c=weights[n], s=0.5, alpha=0.5, label="Band: " + band) 
val = np.moveaxis(val, -1, 0) # [Vertex][Band]
weights = np.moveaxis(weights, 0, 1) # [Vertex][i][j]

plt.ylabel("E - E$_{cbm}$ [eV]")

plot_transitions = np.array([0])
plot_transitions = np.append(plot_transitions, plot_transitions[-1] + Kx_RG.__len__() - 1)
plot_transitions = np.append(plot_transitions, plot_transitions[-1] + Kx_GX.__len__() - 1)
plot_transitions = np.append(plot_transitions, plot_transitions[-1] + Kx_XM.__len__() - 1)
plot_transitions = np.append(plot_transitions, plot_transitions[-1] + Kx_MG.__len__() - 1)

plt.xticks(plot_transitions, ['R', 'Γ', 'X', 'M', 'Γ'])
for plot_transition in plot_transitions:
    plt.axvline(x=plot_transition, color='k', linewidth=0.5, linestyle="-")

plt.legend(["Band: dzy", "Band: dxz", "Band: dxy"])
debug_times = np.append(debug_times, time.time() - start)
debug_labels = np.append(debug_labels, "Plotting")
print("Plot created in " + str(debug_times[-1]))


print("Total time: " + str(time.time() - start_time))
plt.show()


print("Showing Debug graph...")
plt.pie(debug_times, labels = debug_labels, autopct='%1.1f%%')
plt.show()