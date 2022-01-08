import numpy as np
import pandas as pd
import tifffile
import gdal
import utm
import math
from netCDF4 import Dataset

def read_netCDF_data(file_name, lon, lat, out_var):
    z = 0

    # ### Reading tif file

    # im = tifffile.imread(file_name)
    # print(imarray.shape)
    # print(im.shape)

    # ds = gdal.Open(file_name)
    # width = ds.RasterXSize
    # height = ds.RasterYSize
    # gt = ds.GetGeoTransform()
    # minx = gt[0]
    # miny = gt[3] + width*gt[4] + height*gt[5] 
    # maxx = gt[0] + width*gt[1] + height*gt[2]
    # maxy = gt[3]
    # # print(minx, miny)
    # lat_arr = np.linspace(miny, maxy, im.shape[0])
    # lon_arr = np.linspace(minx, maxx, im.shape[1])
    # # print(lat_arr, lon_arr)
    # # print(lat, lon)

    # #### For UTM
    # u = utm.from_latlon(lat, lon, force_zone_number=43)
    # lat = u[1]
    # lon = u[0]
    # # print(u)
    # # print(lat, lon)

    ##### Read netCDF
    read_data=Dataset(file_name,'r')
    var=list(read_data.variables.keys())

    read_lat = read_data.variables['lat'][:]
    read_lon = read_data.variables['lon'][:]
    im = read_data.variables[out_var][:]

    lon_arr = read_lon[:]
    lat_arr = read_lat[:]

    diff_arr_x = np.absolute(lon_arr-lon)
    id_x = diff_arr_x.argmin() # find the index of minimum element from the array

    diff_arr_y = np.absolute(lat_arr-lat)
    id_y = diff_arr_y.argmin() # find the index of minimum element from the array

    # print(id_x, id_y)

    z = im[id_y][id_x]
    # print(z)

    return z

def write_csv(file_name, data_arr):
    with open(file_name, 'w') as csv_file:
        for i in range(len(data_arr)):
            str_write = ''
            for j in range(len(data_arr[i])):
                if(j < len(data_arr[i])-1):
                    str_write += str(data_arr[i][j]) + ','
                else:
                    str_write += str(data_arr[i][j]) + '\n'
            csv_file.write(str_write)
    return 0

def read_data_create_array(Basin_name, input_layers):

    path_coords_read = 'E:\IIT_GN\Academics\Sem_7\CE_499\code\Data\\'+str(Basin_name)+'\\training_data\\data.csv'
    path_input_layers1 = 'E:\IIT_GN\Academics\Sem_7\CE_499\code\Data\\'+str(Basin_name)+'\\input_layers\\netCDF'
    path_input_layers2 = 'E:\IIT_GN\Academics\Sem_7\CE_499\\code\\Data\\LULC_Soil\\'

    matrix_coords = pd.read_csv(path_coords_read, header=None).to_numpy()
    matrix_coords = matrix_coords.astype('float64')
    # print(matrix_coords)
    matrix_coords = matrix_coords[matrix_coords[:, 2].argsort()]
    # print(matrix_coords)

    mode_val = [1]
    mode_itr = 0
    prev_val = matrix_coords[0,2] 
    for i in range(1, len(matrix_coords)):
        if(matrix_coords[i,2] == prev_val):
            mode_val[mode_itr] += 1
        else:
            mode_itr += 1
            prev_val = matrix_coords[i,2]
            mode_val.append(1)
    # print(mode_val)

    validation_data_points = np.round(np.multiply(mode_val, 0.2), decimals=0).astype('int')
    training_data_points = np.subtract(mode_val, validation_data_points)
    # print(validation_data_points, training_data_points)

    ### Randomize points
    randomize_itr_start = 0
    for i in range(len(mode_val)):
        randomize_itr_stop  = int(int(mode_val[i]) + randomize_itr_start)
        # print(randomize_itr_start, randomize_itr_stop)
        # print(matrix_coords[randomize_itr_start:randomize_itr_stop])
        np.random.shuffle(matrix_coords[randomize_itr_start:randomize_itr_stop])
        # print(matrix_coords)
        randomize_itr_start += int(mode_val[i])
    # print(matrix_coords)
    # print(len(matrix_coords))

    #### Writing matrix for training data
    itr = 0
    training_data = []
    # print(len(training_data_points))
    for i in range(len(training_data_points)):
        for j in range(training_data_points[i]):
            data = []
            data.append(matrix_coords[itr+j,0])
            data.append(matrix_coords[itr+j,1])
            data.append(int(matrix_coords[itr+j,2]))
            for k in range(len(input_layers)):
            # for k in range(1):
                path_input_layers = path_input_layers1
                out_var = 'Band1'
                if(k == 10):
                    path_input_layers = path_input_layers2
                    out_var = 'Band1'
                if(k == 11):
                    path_input_layers = path_input_layers2
                    out_var = 'LULC'
                # file_name = path_input_layers + '\\' + input_layers[k] + '.netCDF' # read netCDF file
                file_name = path_input_layers + '\\' + input_layers[k] + '.nc' # read netCDF file
                z = read_netCDF_data(file_name, matrix_coords[itr+j,0], matrix_coords[itr+j, 1], out_var)
                data.append(z)
            # data = np.array(data)
            # data[:, 3:len(data[1])].astype('float64')
            
            data = np.array(data)
            data = np.ma.filled(data,-9999)
            for k in range(2,len(data)):
                if(np.isnan(data[k]) == True):
                    data[k] = -9999
                try:
                    data[k] = data[k].astype('float64')
                except:
                    data[k] = -9999
            # print(data)
            training_data.append(data)
            
        itr += mode_val[i]
    np.random.shuffle(training_data)
    # print(training_data)

    #### Writing matrix for validation data
    itr = training_data_points[0]
    validation_data = []
    # print(len(training_data_points))
    for i in range(len(validation_data_points)):
        for j in range(validation_data_points[i]):
            data = []
            data.append(matrix_coords[itr+j,0])
            data.append(matrix_coords[itr+j,1])
            data.append(int(matrix_coords[itr+j,2]))
            for k in range(len(input_layers)):
                path_input_layers = path_input_layers1
                out_var = 'Band1'
                if(k == 10):
                    path_input_layers = path_input_layers2
                    out_var = 'Band1'
                if(k == 11):
                    path_input_layers = path_input_layers2
                    out_var = 'LULC'
            # for k in range(1):
                # file_name = path_input_layers + '\\' + input_layers[k] + '.tif' # read tif file
                file_name = path_input_layers + '\\' + input_layers[k] + '.nc' # read netCDF file
                z = read_netCDF_data(file_name, matrix_coords[itr+j,0], matrix_coords[itr+j, 1], out_var)
                data.append(z)
            # print(data)
            data = np.array(data)
            data = np.ma.filled(data,-9999)
            for k in range(2,len(data)):
                if(np.isnan(data[k]) == True):
                    data[k] = -9999
                try:
                    data[k] = data[k].astype('float64')
                except:
                    data[k] = -9999
            # print(data)
            validation_data.append(data)
        itr += mode_val[i]
    np.random.shuffle(validation_data)
    # print(validation_data)

    ####### write files
    path_save = 'E:\IIT_GN\Academics\Sem_7\CE_499\code\Processed_data\\'+str(Basin_name)+'\\'
    write_csv(path_save+Basin_name+'_training.csv', training_data)
    write_csv(path_save+Basin_name+'_validation.csv', validation_data)

basins = ['Brahmaputra']
input_layers = ['Filled_DEM', 'Slope', 'Aspect', 'Total_Curvature', 'TRI', 'TWI', 
    'SPI', 'STI', 'Precipitation', 'Dist_Stream', 'soil_classification', 'LULC_netCDF', 'Streamflow']

for i in range(len(basins)):
    read_data_create_array(basins[i], input_layers)