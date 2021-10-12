import numpy as np

def read_files(file_name):

    """
    Reads in data from the file_names provided
    into the data attribute. Data must be in the form:
        Rx,Ry,Rz,i,j,tr,ti
    Files are closed after they are read.

    Returns
    -------
    None.

    """
    raw_data = np.zeros((0,6))

    with open(file_name, mode = "r") as file: #Will allways close file
        lines = file.readlines()[92:]
        for line in lines:
            values = line.split()
            raw_data = np.vstack((raw_data, 
                np.array([int(values[0]), int(values[1]), int(values[2]),int(values[3]), int(values[4]), complex(float(values[5]), float(values[6]))]
                )))
    
    return raw_data

print("Inporting...")
num_bands = 3
num_vertices = 1331
raw_data = read_files("SrTiO3_hr.dat")
print("Impoort complete")

print("Splitting data...")


print("Building Hamaltonian")
k = (1, 0, 0)
h = 0 + 0j
for line in raw_data:
    h =+ line[5] * np.exp(1j * (k[0] * line[0] + k[1] * line[1] + k[2] * line[2]))
print(h)