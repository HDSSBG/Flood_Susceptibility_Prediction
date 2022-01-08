import numpy as np
import pandas as pd
from netCDF4 import Dataset
#from pandas.core.dtypes.missing import isnull
from pysheds.grid import Grid
#from pysheds.pgrid import Grid as Grid

def read_netCDF_data(prec_file, id_x, id_y, out_var):
    z = 0

    ##### Read netCDF
    read_data=Dataset(prec_file,'r')
    var=list(read_data.variables.keys())

    read_lat = read_data.variables['lat'][:]
    #print(read_lat)
    #read_lat = read_lat[:]
    read_lon = read_data.variables['lon'][:]
    #print(read_lon)
    im = read_data.variables[out_var][:]
    #print(im)

    lon_arr = read_lon[:]
    lat_arr = read_lat[:]

    # diff_arr_x = np.absolute(lon_arr-lon)
    # id_x = diff_arr_x.argmin() # find the index of minimum element from the array

    # diff_arr_y = np.absolute(lat_arr-lat)
    # id_y = diff_arr_y.argmin() # find the index of minimum element from the array

    # print(id_x, id_y)

    z = im[id_y][id_x]
    #print(z)
    # print(z)

    return z

def find_acc_prec(final_list):
    prec_file = "E:\\IIT_GN\\Academics\\Sem_7\\CE_499\\code\\Data\\Brahmaputra\\input_layers\\netCDF\\Precipitation.nc"
    out_var = 'Band1'

    acc_prec = 0
    for i in range(len(final_list)):
        for j in range(len(final_list[i])):
            if(np.isnan(final_list[i][j]) == False and int(final_list[i][j]) == 1):
                val = read_netCDF_data(prec_file, j, i, out_var)
                if(np.isnan(val)==False):
                    val = float(val)
                    acc_prec += val
                
    return acc_prec

def delineation(dem_file, pour_loc):
    grid = Grid.from_raster(dem_file)
    dem = grid.read_raster(dem_file)

    # Fill depressions in DEM
    flooded_dem = grid.fill_depressions(dem)
    
    # Resolve flats in DEM
    inflated_dem = grid.resolve_flats(flooded_dem)

    # Specify directional mapping
    dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
    
    # Compute flow directions
    fdir = grid.flowdir(inflated_dem, dirmap=dirmap)

    # flow accumulation
    acc = grid.accumulation(fdir, dirmap=dirmap)

    # Delineate a catchment
    # Specify pour point
    x_pour, y_pour = pour_loc[0], pour_loc[1]

    # Snap pour point to high accumulation cell
    #x_pour, y_pour = grid.snap_to_mask(acc > 10, (x_pour, y_pour))

    # Delineate the catchment
    catch = grid.catchment(x=x_pour, y=y_pour, fdir=fdir, dirmap=dirmap, 
                            xytype='coordinate')

    # Crop and plot the catchment
    # ---------------------------
    # Clip the bounding box to the catchment
    # grid.clip_to(catch)
    # catch = grid.view(catch)

    final_list = np.where(catch, catch, np.nan)
    #print(final_list)

    acc_rainfall = find_acc_prec(final_list)

    #print(acc_rainfall)

    return acc_rainfall

file_path = 'E:\\IIT_GN\\Academics\\Sem_7\\CE_499\\code\\Data\\Brahmaputra\\input_layers\\netCDF\\'
dem_data = 'Filled_DEM.tif'
Basin_name = 'Brahmaputra'

dem_file = file_path+dem_data

path_coords_read = 'E:\IIT_GN\Academics\Sem_7\CE_499\\code\\Data\\'+str(Basin_name)+'\\grids\\'+str(Basin_name)+'_grid.csv'

matrix_coords = pd.read_csv(path_coords_read, header=1).to_numpy()
matrix_coords = matrix_coords.astype('float64')
# print(matrix_coords)
matrix_coords = matrix_coords[matrix_coords[:, 1].argsort()]

prev_file = "E:\\IIT_GN\\Academics\\Sem_7\\CE_499\\code\\Data\\Brahmaputra\\input_layers\\netCDF\\Precipitation.nc"
read_data=Dataset(prev_file,'r')
var=list(read_data.variables.keys())
read_lat = read_data.variables['lat'][:]
read_lon = read_data.variables['lon'][:]

lonrect = read_lon[:]
latrect = read_lat[:]

l = np.full((len(latrect), len(lonrect)), None)


for i in range(len(matrix_coords)):
    pour_loc = [matrix_coords[i,0], matrix_coords[i,1]]

    diff_arr_x = np.absolute(lonrect-matrix_coords[i,0])
    id_x = diff_arr_x.argmin() # find the index of minimum element from the array

    diff_arr_y = np.absolute(latrect-matrix_coords[i,1])
    id_y = diff_arr_y.argmin() # find the index of minimum element from the array

    temp_val = delineation(dem_file, pour_loc)
    l[id_y, id_x] = temp_val

path_save = 'E:\\IIT_GN\\Academics\\Sem_7\\CE_499\\code\\Data\\Brahmaputra\\input_layers\\netCDF\\'

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