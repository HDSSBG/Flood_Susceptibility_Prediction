import numpy as np
import pandas as pd
from netCDF4 import Dataset
#from pandas.core.dtypes.missing import isnull
from pysheds.grid import Grid
#from pysheds.pgrid import Grid as Grid

def read_netCDF_data(lat, lon, lon_arr, lat_arr, im):
    z = 0

    ##### Read netCDF
    

    diff_arr_x = np.absolute(lon_arr-lon)
    id_x = diff_arr_x.argmin() # find the index of minimum element from the array

    diff_arr_y = np.absolute(lat_arr-lat)
    id_y = diff_arr_y.argmin() # find the index of minimum element from the array

    # print(id_x, id_y)

    z = im[id_y][id_x]
    #print(z)
    # print(z)

    return z

# def find_acc_prec(im, lat_arr, lon_arr, im_ele, lat_arr_ele, lon_arr_ele):

#     acc_prec = 0
#     final_list = np.array(im_ele)
#     results = np.where(final_list == 1)
#     listOfCoordinates= list(zip(results[0], results[1]))

#     for coords in listOfCoordinates:
#         lat = lat_arr_ele[coords[0]]
#         lon = lon_arr_ele[coords[1]]
#         # print(lat, lon)
#         val = read_netCDF_data(lat, lon, lon_arr, lat_arr, im)
#         if(np.isnan(val)==False):
#             val = float(val)
#             acc_prec += val
                
#     return acc_prec

def find_acc_prec1(loc, im, lat_arr, lon_arr, im_ele, lat_arr_ele, lon_arr_ele):
    val = 0

    x = read_netCDF_data(loc[1], loc[0], lon_arr_ele, lat_arr_ele, im_ele)

    if(x == 1):
        val = read_netCDF_data(loc[1], loc[0], lon_arr, lat_arr, im)

    return val

file_path = 'E:\IIT_GN\Academics\Sem_8\CE_499\Single_basin\Single_Basin\model_codes\Ver_1\code\Data\Brahmaputra\subcatchment\\netCDF\\'
Basin_name = 'Brahmaputra'
out_var = 'Band1'

path_coords_read = 'E:\IIT_GN\Academics\Sem_8\CE_499\Single_basin\Single_Basin\model_codes\Ver_1\code\Data\\'+str(Basin_name)+'\\grids\\'+str(Basin_name)+'_grid.csv'

matrix_coords = pd.read_csv(path_coords_read, header=1).to_numpy()
matrix_coords = matrix_coords.astype('float64')
matrix_coords = matrix_coords[matrix_coords[:, 1].argsort()]

prec_file = "E:\IIT_GN\Academics\Sem_8\CE_499\Single_basin\Single_Basin\model_codes\Ver_1\code\Data\Brahmaputra\\input_layers\\netCDF\\Precipitation.nc"
read_data=Dataset(prec_file,'r')
var=list(read_data.variables.keys())

read_lat = read_data.variables['lat'][:]
read_lon = read_data.variables['lon'][:]
im = read_data.variables[out_var][:]
#print(im)

lon_arr = read_lon[:]
lat_arr = read_lat[:]

scaling = (0.01/0.25)**2

lonrect = read_lon[:]
latrect = read_lat[:]

l = np.full((len(latrect), len(lonrect)), None)


# for i in range(len(matrix_coords)):
#     pour_loc = [matrix_coords[i,0], matrix_coords[i,1]]

#     diff_arr_x = np.absolute(lonrect-matrix_coords[i,0])
#     id_x = diff_arr_x.argmin() # find the index of minimum element from the array

#     diff_arr_y = np.absolute(latrect-matrix_coords[i,1])
#     id_y = diff_arr_y.argmin() # find the index of minimum element from the array

#     prev_ele_file = "E:\IIT_GN\Academics\Sem_8\CE_499\Single_basin\Single_Basin\model_codes\Ver_1\code\Data\Brahmaputra\subcatchment\\netCDF\\subcatchment_"+str(i)+".nc"

#     read_data=Dataset(prev_ele_file,'r')
#     var=list(read_data.variables.keys())

#     read_lat = read_data.variables['lat'][:]
#     read_lon = read_data.variables['lon'][:]
#     im_ele = read_data.variables[out_var][:]

#     lon_arr_ele = read_lon[:]
#     lat_arr_ele = read_lat[:]

#     temp_val = find_acc_prec(im, lat_arr, lon_arr, im_ele, lat_arr_ele, lon_arr_ele)*scaling
#     l[id_y, id_x] = temp_val

for i in range(len(matrix_coords)):

    diff_arr_x = np.absolute(lonrect-matrix_coords[i,0])
    id_x = diff_arr_x.argmin() # find the index of minimum element from the array

    diff_arr_y = np.absolute(latrect-matrix_coords[i,1])
    id_y = diff_arr_y.argmin() # find the index of minimum element from the array

    prev_ele_file = "E:\IIT_GN\Academics\Sem_8\CE_499\Single_basin\Single_Basin\model_codes\Ver_1\code\Data\Brahmaputra\subcatchment\\netCDF\\subcatchment_"+str(i)+".nc"

    read_data=Dataset(prev_ele_file,'r')
    var=list(read_data.variables.keys())

    read_lat = read_data.variables['lat'][:]
    read_lon = read_data.variables['lon'][:]
    im_ele = read_data.variables[out_var][:]

    lon_arr_ele = read_lon[:]
    lat_arr_ele = read_lat[:]

    temp_val = 0
    for j in range(len(matrix_coords)):
        loc = [matrix_coords[j,0], matrix_coords[j,1]]
        temp = find_acc_prec1(loc, im, lat_arr, lon_arr, im_ele, lat_arr_ele, lon_arr_ele)
        if(np.isnan(temp) == False):
            temp_val += temp
    
    l[id_y, id_x] = temp_val

path_save = 'E:\\IIT_GN\\Academics\\Sem_8\\CE_499\\Single_basin\\Single_Basin\\model_codes\\Ver_1\\code\Data\\Brahmaputra\\input_layers\\netCDF\\'

ds=Dataset(path_save+'Accumulated_Precipitation.nc', mode="w", format='NETCDF4')
# some file-level meta-data attributes:
ds.Conventions = "CF-1.6" 
ds.title = 'Accumulated Precipitation'


nlat=ds.createDimension('lat', len(latrect))
nlon=ds.createDimension('lon', len(lonrect))


longitude = ds.createVariable("lon","f8",("lon",))
latitude = ds.createVariable("lat","f8",("lat",))
longitude.units = "degrees_east"
latitude.units = "degrees_north"

humr = ds.createVariable("Band1","f8",("lat","lon",),
                            chunksizes=(len(latrect),len(lonrect)), zlib=True, complevel=9)

humr.description = "Precipitation (kg m-2 s-1)"

longitude[:] = lonrect.reshape((len(lonrect),1))
latitude[:] = latrect.reshape((len(latrect),1))
humr[:] = l

ds.close()