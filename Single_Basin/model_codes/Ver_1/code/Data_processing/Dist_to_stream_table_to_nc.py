import numpy as np
from netCDF4 import Dataset
import pandas as pd

def get_rect_array(lat_arr, lon_arr):
    latrect = []
    for i in range(len(lat_arr)):
        if(lat_arr[i] not in latrect):
            latrect.append(lat_arr[i])
    latrect.sort()
    latrect = np.array(latrect)

    lonrect = []
    for i in range(len(lon_arr)):
        if(lon_arr[i] not in lonrect):
            lonrect.append(lon_arr[i])
    lonrect.sort()
    lonrect = np.array(lonrect)

    return [latrect, lonrect]

def extract_data(latrect, lonrect, grid_data, l):
    for i in range(len(grid_data)):
        diff_arr_x = np.absolute(lonrect - grid_data[i,0])
        id_x = diff_arr_x.argmin()

        diff_arr_y = np.absolute(latrect - grid_data[i,1])
        id_y = diff_arr_y.argmin()

        l[id_y, id_x] = grid_data[i,2]
    return l

basin_name = 'Narmada'
input_file_path = 'E:\\IIT_GN\\Academics\\Sem_7\\CE_499\\code\Data\\'+basin_name+'\\grids\\'
path_save = 'E:\\IIT_GN\\Academics\\Sem_7\\CE_499\\code\\Data\\'+basin_name+'\\input_layers\\netCDF\\'

grids_input_file_0 = pd.read_csv(input_file_path+"dist_grid.csv", header=1).to_numpy()
grids_input_file = grids_input_file_0[:,:]
grids_input_file = grids_input_file.astype('float64')

ds=Dataset(path_save+'Dist_Stream.nc', mode="w", format='NETCDF4')
# some file-level meta-data attributes:
ds.Conventions = "CF-1.6" 
ds.title = 'Distance to the Nearest Stream'

[latrect, lonrect] = get_rect_array(grids_input_file[:,1], grids_input_file[:,0])
l = np.full((len(latrect), len(lonrect)), None)
l = extract_data(latrect, lonrect, grids_input_file, l)

nlat=ds.createDimension('lat', len(latrect))
nlon=ds.createDimension('lon', len(lonrect))


longitude = ds.createVariable("lon","f8",("lon",))
latitude = ds.createVariable("lat","f8",("lat",))
longitude.units = "degrees_east"
latitude.units = "degrees_north"
humr = ds.createVariable("Band1","f8",("lat","lon",),
                            chunksizes=(len(latrect),len(lonrect)), zlib=True, complevel=9)

humr.unit = "m"

longitude[:] = lonrect.reshape((len(lonrect),1))
latitude[:] = latrect.reshape((len(latrect),1))
humr[:] = l

ds.close()