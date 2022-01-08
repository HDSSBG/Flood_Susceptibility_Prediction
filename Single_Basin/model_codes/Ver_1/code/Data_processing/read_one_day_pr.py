import numpy as np
from netCDF4 import Dataset
import os
import csv
import pandas as pd
import re
# load the data
def netpr(basin_name):
    pathf = 'E:\\IIT_GN\\Academics\\Sem_7\\CE_499\\code\\Data\\'+str(basin_name)+'\\input_layers'
    
    read_data=Dataset(pathf+'\\pr_'+str(basin_name)+'_from_1941.nc','r')
    var=list(read_data.variables.keys())

    start_year = 1984
    end_year = 2016
    # 1984, 2006

    start_idx = abs(1941 - start_year)*365 + int(abs(1941 - start_year)%4)
    if(end_year%4 == 0):
        end_idx = start_idx + 367
    else:
        end_idx = start_idx + 366

    read_days = read_data.variables['time'][:]
    read_lat = read_data.variables['lat'][:]
    read_lon = read_data.variables['lon'][:]
    read_var = read_data.variables['prec'][start_idx:end_idx,:,:]

    lonrect = read_lon[:]
    latrect = read_lat[:]

    print(lonrect)
    print(latrect)
    days = np.arange(1,len(read_days)+3653,1, dtype=int)

    print(len(days))
    #print(read_var)
    print(read_var.shape)
    

    #print(l)
    finalpath = 'E:\\IIT_GN\\Academics\\Sem_7\\CE_499\\code\\Data\\'+str(basin_name)+'\\input_layers\\netCDF'
    os.chdir(finalpath)
    # this load the file into a Nx3 array (three columns)

    # create a netcdf Data object
    ds=Dataset('Precipitation.nc', mode="w", format='NETCDF4')
        # some file-level meta-data attributes:
    ds.Conventions = "CF-1.6" 
    ds.title = 'Precipitation'
    

    nlat=ds.createDimension('lat', len(latrect))
    nlon=ds.createDimension('lon', len(lonrect))

    
    longitude = ds.createVariable("lon","f8",("lon",))
    latitude = ds.createVariable("lat","f8",("lat",))
    longitude.units = "degrees_east"
    latitude.units = "degrees_north"
   
    humr = ds.createVariable("Band1","f8",("lat","lon",),
                             chunksizes=(len(latrect),len(lonrect)), zlib=True, complevel=9)
    humr.units = "kg m-2 s-1"
        # time = ds.createDimension('time', 0)
    longitude[:] = lonrect.reshape((len(lonrect),1))
    latitude[:] = latrect.reshape((len(latrect),1))
    humr[:] = np.amax(read_var, axis = 0)
    ds.close()
        ## adds some attributes

print(1)
netpr('BRAHMAPUTRA')