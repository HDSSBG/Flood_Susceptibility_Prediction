import numpy as np
from netCDF4 import Dataset
import random

def check_extent(file_name, lat_val, lon_val):
    read_data=Dataset(file_name,'r')
    var=list(read_data.variables.keys())

    read_lat = read_data.variables['lat'][:]
    read_lon = read_data.variables['lon'][:]
    im = read_data.variables['Band1'][:]

    lon_arr = read_lon[:]
    lat_arr = read_lat[:]

    diff_arr_x = np.absolute(lon_arr-lon_val)
    id_x = diff_arr_x.argmin() # find the index of minimum element from the array

    diff_arr_y = np.absolute(lat_arr-lat_val)
    id_y = diff_arr_y.argmin()

    x = im[id_y][id_x]

    try:
        x = int(x)
    except:
        x = 0

    #print(x)

    return x

def read_netCDF_data(file_name, extent_file, output_file):
    num_cells = []
    num_each_class = 87

    ##### Read netCDF
    read_data=Dataset(file_name,'r')
    var=list(read_data.variables.keys())

    read_lat = read_data.variables['lat'][:]
    read_lon = read_data.variables['lon'][:]
    im = read_data.variables['Band1'][:]

    lon_arr = read_lon[:]
    lat_arr = read_lat[:]

    lon_positive = []
    num_positive = 0
    lon_negative = []
    num_negative = 0

    for i in range(len(lat_arr)):        
        lon_positive.append([])
        lon_negative.append([])
        for j in range(len(lon_arr)):
            temp_var = im[i][j]
            x = check_extent(extent_file, lat_arr[i], lon_arr[j])
            if(np.isnan(temp_var) == False and int(temp_var) == 1 and x == 1):
                lon_positive[i].append(lon_arr[j])
                num_positive += 1
            elif(np.isnan(temp_var) == False and int(temp_var) == 0 and x == 1):
                lon_negative[i].append(lon_arr[j])
                num_negative += 1

    num_each_lat_positive = []
    num_each_lat_negative = []

    #print(num_positive)
    #print(num_negative)

    coeff_pos = 1
    while(sum(num_each_lat_positive) < num_each_class - 5 or sum(num_each_lat_positive)>num_each_class):
        num_each_lat_positive = []
        for i in range(len(lon_positive)):
            #print(len(lon_positive[i]))
            num_each_lat_positive.append(int(len(lon_positive[i])*num_each_class*coeff_pos/num_positive))
        if(sum(num_each_lat_positive) == 0):
            coeff_pos += 1
        else:
            coeff_pos += (num_each_class - sum(num_each_lat_positive)) / sum(num_each_lat_positive)
        #print(sum(num_each_lat_positive))

    coeff_neg = 1
    while(sum(num_each_lat_negative) < num_each_class  - 5 or sum(num_each_lat_negative)>num_each_class):
        num_each_lat_negative = []
        for i in range(len(lon_negative)):
            #print(len(lon_negative[i]))
            num_each_lat_negative.append(int(len(lon_negative[i])*num_each_class*coeff_neg/num_negative))
        if(sum(num_each_lat_negative) == 0):
            coeff_neg += 1
        else:
            coeff_neg += (num_each_class - sum(num_each_lat_negative)) / sum(num_each_lat_negative)
    #print(num_each_lat_negative)
    
    
    num_each_lat_positive[num_each_lat_positive.index(max(num_each_lat_positive))] += num_each_class - sum(num_each_lat_positive)
    num_each_lat_negative[num_each_lat_negative.index(max(num_each_lat_negative))] += num_each_class - sum(num_each_lat_negative)

    data_pos = []
    data_neg = []

    for i in range(len(lat_arr)):
        #selection_arr = np.linspace(0, len(lon_positive[i])-1, num_each_lat_positive[i], endpoint = True)
        arr_idx = list(np.arange(0, len(lon_positive[i])))
        selection_arr = random.sample(arr_idx, num_each_lat_positive[i])
        #print(selection_arr)
        #print(len(lon_positive[i]))
        for j in range(len(selection_arr)):    
            data_pos.append([lat_arr[i], lon_positive[i][int(selection_arr[j])]])

    for i in range(len(lat_arr)):
        # if(num_each_lat_negative[i] != 0):
        #     increment_factor = len(lon_negative[i])/num_each_lat_negative[i]
        #     #selection_arr = np.linspace(0, len(lon_negative[i])-1, num_each_lat_negative[i], endpoint = False)
        #     selection_arr = np.arange(increment_factor, len(lon_negative[i])+1, increment_factor)
        # #print(len(lon_negative[i])-1)
        # else:
        #     selection_arr = []
        arr_idx = list(np.arange(0, len(lon_negative[i])))
        selection_arr = random.sample(arr_idx, num_each_lat_negative[i])
        #print(selection_arr)
        for j in range(len(selection_arr)):    
            data_neg.append([lat_arr[i], lon_negative[i][int(selection_arr[j])]])

    #print(data_pos, data_neg)

    write_csv(data_pos, data_neg, output_file)

    return 0

def write_csv(data_pos, data_neg, output_file):
    with open(output_file, 'w') as csv_file:
        str_write = ''
        for i in range(len(data_pos)):
            str_write += str(data_pos[i][1])+','+str(data_pos[i][0])+','+str(1)+'\n'
        for i in range(len(data_neg)):
            str_write += str(data_neg[i][1])+','+str(data_neg[i][0])+','+str(0)+'\n'
        csv_file.write(str_write)
    return 0

file_path = 'E:\IIT_GN\Academics\Sem_7\CE_499\code\Data\Brahmaputra\\training_data\\'
input_file = file_path+'Brahmaputra_Total_Flooded.nc'
extent_file = file_path+'Brahmaputra_Area.nc'
output_file = file_path+'data.csv'

read_netCDF_data(input_file, extent_file, output_file)