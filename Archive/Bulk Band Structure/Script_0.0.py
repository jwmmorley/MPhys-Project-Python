import numpy as np
import pandas as pd


def read_line(line, delimiter = "    ", type = int):

    ret = np.array([])
    values = line.split(delimiter)
    for i in range(0, values.__len__()):

        if isinstance(values[i], type):
            np.append(ret, int(values[i]))

    print (ret)
    return ret, ret.__len__()

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
    with open(file_name, mode = "r") as file: #Will allways close file
        file.readline()
        num_bands = int(file.readline())
        num_vectors = int(file.readline())

        weights = np.array([])
        n = 0
        while n < 100:
            line = file.readline()
            print(line)
            np.append(weights, read_line(line))
            n  =+ line.__len__()

        #data = np.zeros((0,7))
        #temp_data = np.genfromtxt(file, dtype = float, comments = "%",
        #        delimiter = "\t", skip_header=1)
        #data = np.vstack((data, temp_data))

        data = weights
    
        
    return data, num_bands, num_vectors

data, _, _ = read_files("SrTiO3_hr.dat")
print(data)


weight_of_R_vector_df = pd.DataFrame()
"""
with open('datafile.txt') as datafile:
    weight_of_R_vector_raw = datafile.readlines()[3:92]
for line in weight_of_R_vector_raw:
    line_delimited = re.sub("\s+", ",", line.strip())
    print(line_delimited)
    weight_of_R_vector_row = pd.DataFrame(data=np.array([['Dev','comp',100]]), columns=['Name','Subj','Marks'])
    weight_of_R_vector_df = pd.concat([weight_of_R_vector_df,weight_of_R_vector_row], ignore_index=True)

weight_of_R_vector_df 
"""
lines = ''
with open('datafile.txt') as datafile:
    weight_of_R_vector_raw = datafile.readlines()[3:92]
for line in weight_of_R_vector_raw:
    line_delimited = re.sub("\s+", ",", line.strip())
    #print(line_delimited)
    lines = lines + ',' + line_delimited 

datafile.close()
numbers = re.findall('[0-9]+', lines)

weight_of_R_vector_df = pd.DataFrame(numbers, columns = ['Weight of R vector'])
del weight_of_R_vector_raw

weight_of_R_vector_df