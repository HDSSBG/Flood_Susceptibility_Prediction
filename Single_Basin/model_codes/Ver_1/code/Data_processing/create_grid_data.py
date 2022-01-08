import numpy as np
import pandas as pd
from netCDF4 import Dataset
from pandas.core.dtypes.missing import isnull

def read_netCDF_data(file_name, lon, lat, out_var):
    z = 0

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

    path_coords_read = 'E:\IIT_GN\Academics\Sem_7\CE_499\\code\\Data\\'+str(Basin_name)+'\\grids\\'+str(Basin_name)+'_grid.csv'
    path_input_layers1 = 'E:\IIT_GN\Academics\Sem_7\CE_499\code\Data\\'+str(Basin_name)+'\\input_layers\\netCDF'
    path_input_layers2 = 'E:\IIT_GN\Academics\Sem_7\CE_499\\code\\Data\\LULC_Soil\\'

    matrix_coords = pd.read_csv(path_coords_read, header=1).to_numpy()
    matrix_coords = matrix_coords.astype('float64')
    # print(matrix_coords)
    matrix_coords = matrix_coords[matrix_coords[:, 1].argsort()]
    # print(matrix_coords)

    # print(validation_data_points, training_data_points)

    matrix_data = []
    # print(len(training_data_points))
    for i in range(len(matrix_coords)):
        data = []
        data.append(matrix_coords[i,0])
        data.append(matrix_coords[i,1])
        for k in range(len(input_layers)):
            path_input_layers = path_input_layers1
            out_var = 'Band1'
            if(k == 10):
                path_input_layers = path_input_layers2
                out_var = 'Band1'
            if(k == 11):
                path_input_layers = path_input_layers2
                out_var = 'LULC'
            file_name = path_input_layers + '\\' + input_layers[k] + '.nc' # read netCDF file
            z = read_netCDF_data(file_name, matrix_coords[i,0], matrix_coords[i, 1], out_var)
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
        # data = np.array(data)
        # data[:, 3:len(data[1])].astype('float64')
        matrix_data.append(data)
        # print(i)
    # print(training_data)


    ####### write files
    path_save = 'E:\IIT_GN\Academics\Sem_7\CE_499\code\Processed_data\\'+str(Basin_name)+'\\grids\\'
    write_csv(path_save+'grid_input.csv', matrix_data)

basins = ['Brahmaputra']
input_layers = ['Filled_DEM', 'Slope', 'Aspect', 'Total_Curvature', 'TRI', 'TWI', 
    'SPI', 'STI', 'Precipitation', 'Dist_Stream', 'soil_classification', 'LULC_netCDF', 'Streamflow']

for i in range(len(basins)):
    read_data_create_array(basins[i], input_layers)