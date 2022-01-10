import numpy as np
import pandas as pd
from netCDF4 import Dataset
#from pandas.core.dtypes.missing import isnull
from pysheds.grid import Grid
#from pysheds.pgrid import Grid as Grid

# def read_netCDF_data(prec_file, id_x, id_y, out_var):
#     z = 0

#     ##### Read netCDF
#     read_data=Dataset(prec_file,'r')
#     var=list(read_data.variables.keys())

#     read_lat = read_data.variables['lat'][:]
#     #print(read_lat)
#     #read_lat = read_lat[:]
#     read_lon = read_data.variables['lon'][:]
#     #print(read_lon)
#     im = read_data.variables[out_var][:]
#     #print(im)

#     lon_arr = read_lon[:]
#     lat_arr = read_lat[:]

#     # diff_arr_x = np.absolute(lon_arr-lon)
#     # id_x = diff_arr_x.argmin() # find the index of minimum element from the array

#     # diff_arr_y = np.absolute(lat_arr-lat)
#     # id_y = diff_arr_y.argmin() # find the index of minimum element from the array

#     # print(id_x, id_y)

#     z = im[id_y][id_x]
#     #print(z)
#     # print(z)

#     return z

# def find_acc_prec(final_list):
#     prec_file = "E:\\IIT_GN\\Academics\\Sem_8\\CE_499\\Single_basin\\Single_Basin\\model_codes\\Ver_1\\code\\Data\\Brahmaputra\\input_layers\\netCDF\\Precipitation.nc"
#     out_var = 'Band1'

#     acc_prec = 0
#     for i in range(len(final_list)):
#         for j in range(len(final_list[i])):
#             if(np.isnan(final_list[i][j]) == False and int(final_list[i][j]) == 1):
#                 val = read_netCDF_data(prec_file, j, i, out_var)
#                 if(np.isnan(val)==False):
#                     val = float(val)
#                     acc_prec += val
                
#     return acc_prec

def delineation(dem_file):
    grid = Grid.from_raster(dem_file)
    # print(grid)
    dem = grid.read_raster(dem_file)

    # Fill depressions in DEM
    flooded_dem = grid.fill_depressions(dem)
    
    # Resolve flats in DEM
    inflated_dem = grid.resolve_flats(flooded_dem)

    # Specify directional mapping
    dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
    
    # Compute flow directions
    fdir = grid.flowdir(inflated_dem, dirmap=dirmap)
    #fdir = grid.flowdir(dem, dirmap=dirmap)

    # flow accumulation
    acc = grid.accumulation(fdir, dirmap=dirmap)

    print(np.amax(acc))

    return grid, fdir, dirmap, acc
    # Delineate a catchment
    # Specify pour point
    

def catchment(grid, fdir, dirmap, acc, pour_loc, half_scale_row, flag):
    x, y = pour_loc[0], pour_loc[1]
    #print(x,y)
    col, row = grid.nearest_cell(x, y)
    #print(col, row)

    col_max, row_max = col+half_scale_row, row+half_scale_row
    if(flag == 1):
        col_min, row_min = col-half_scale_row-1, row-half_scale_row-1
    else:
        col_min, row_min = col-half_scale_row, row-half_scale_row

    #print(row_min, row_max, col_min, col_max)
    sub_acc = acc[row_min:row_max+1, col_min:col_max+1]
    #print(sub_acc)
    max_acc = np.amax(sub_acc)
    #print(max_acc)

    # Snap pour point to high accumulation cell
    x_pour, y_pour = grid.snap_to_mask(acc > max_acc-1, (x, y))

    # Delineate the catchment
    catch = grid.catchment(x=x_pour, y=y_pour, fdir=fdir, dirmap=dirmap, 
                            xytype='coordinate')

    # Crop and plot the catchment
    # ---------------------------
    # Clip the bounding box to the catchment
    # grid.clip_to(catch)
    # catch = grid.view(catch)

    final_list = np.where(catch, catch, np.nan)

    final_list = np.flip(final_list, axis = 0)
    #print(final_list)

    #acc_rainfall = find_acc_prec(final_list)

    #print(acc_rainfall)

    return final_list, max_acc

file_path = 'E:\\IIT_GN\\Academics\\Sem_8\\CE_499\\Single_basin\\Single_Basin\\model_codes\\Ver_1\\code\\Data\\Brahmaputra\\input_layers\\netCDF\\'
dem_data = 'Filled_DEM_WGS_0.01.tif'
#dem_data = 'Filled_DEM.tif'
Basin_name = 'Brahmaputra'

dem_file = file_path+dem_data

path_coords_read = 'E:\\IIT_GN\\Academics\\Sem_8\\CE_499\\Single_basin\\Single_Basin\\model_codes\\Ver_1\\code\\Data\\'+str(Basin_name)+'\\grids\\'+str(Basin_name)+'_grid.csv'

matrix_coords = pd.read_csv(path_coords_read, header=1).to_numpy()
matrix_coords = matrix_coords.astype('float64')
# print(matrix_coords)
matrix_coords = matrix_coords[matrix_coords[:, 1].argsort()]

prev_file = "E:\\IIT_GN\\Academics\\Sem_8\\CE_499\\Single_basin\\Single_Basin\\model_codes\\Ver_1\\code\\Data\\Brahmaputra\\input_layers\\netCDF\\Filled_DEM_WGS_0.01.nc"
#prev_file = "E:\\IIT_GN\\Academics\\Sem_8\\CE_499\\Single_basin\\Single_Basin\\model_codes\\Ver_1\\code\\Data\\Brahmaputra\\input_layers\\netCDF\\Filled_DEM.nc"
read_data=Dataset(prev_file,'r')
var=list(read_data.variables.keys())
read_lat = read_data.variables['lat'][:]
read_lon = read_data.variables['lon'][:]

lonrect = read_lon[:]
latrect = read_lat[:]

half_scale_row = int(0.25/(0.01*2))
half_scale_float = 0.25/(0.01*2)
# half_scale_row = int(0.25/(0.25*2))
# half_scale_float = 0.25/(0.25*2)
flag = 0
if(half_scale_row < half_scale_float and half_scale_row != 0):
    flag = 1
#print(half_scale_row)

l = np.full((len(latrect), len(lonrect)), None)

path_save = 'E:\\IIT_GN\\Academics\\Sem_8\\CE_499\\Single_basin\\Single_Basin\\model_codes\\Ver_1\\code\\Data\\Brahmaputra\\subcatchment\\netCDF\\'
#path_save = 'E:\\IIT_GN\\Academics\\Sem_8\\CE_499\\Single_basin\\Single_Basin\\model_codes\\Ver_1\\code\\Data\\Brahmaputra\\subcatchment_old\\netCDF\\'

grid, fdir, dirmap, acc = delineation(dem_file)

for i in range(len(matrix_coords)):
#for i in range(2):

    l = np.full((len(latrect), len(lonrect)), None)

    pour_loc = [matrix_coords[i,0], matrix_coords[i,1]]
    #print(pour_loc)

    # diff_arr_x = np.absolute(lonrect-matrix_coords[i,0])
    # id_x = diff_arr_x.argmin() # find the index of minimum element from the array

    # diff_arr_y = np.absolute(latrect-matrix_coords[i,1])
    # id_y = diff_arr_y.argmin() # find the index of minimum element from the array

    temp_val, max_acc = catchment(grid, fdir, dirmap, acc, pour_loc, half_scale_row, flag)
    l = temp_val

    if(max_acc > 30000):
        print(str(i)+" "+str(max_acc))

    ds=Dataset(path_save+'subcatchment_'+str(i)+'.nc', mode="w", format='NETCDF4')
    # some file-level meta-data attributes:
    ds.Conventions = "CF-1.6" 
    ds.title = 'Subcatchment Area'


    nlat=ds.createDimension('lat', len(latrect))
    nlon=ds.createDimension('lon', len(lonrect))


    longitude = ds.createVariable("lon","f8",("lon",))
    latitude = ds.createVariable("lat","f8",("lat",))
    longitude.units = "degrees_east"
    latitude.units = "degrees_north"

    humr = ds.createVariable("Band1","f8",("lat","lon",),
                                chunksizes=(len(latrect),len(lonrect)), zlib=True, complevel=9)

    humr.description = "Subcatchement"

    longitude[:] = lonrect.reshape((len(lonrect),1))
    latitude[:] = latrect.reshape((len(latrect),1))
    humr[:] = l

    ds.close()

    #print(str(i), end=" ")