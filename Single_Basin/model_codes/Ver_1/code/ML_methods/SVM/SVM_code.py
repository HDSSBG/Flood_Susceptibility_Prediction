import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pandas as pd
#from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt 
from netCDF4 import Dataset
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

# from info_gain import info_gain

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

basin_name = 'Brahmaputra'
input_path = 'C:\\Users\\Dell\\Ver_0_hari1\\code\\Processed_data\\'
# from sklearn.gaussian_process import GaussianProcessClassifier
# from sklearn.gaussian_process.kernels import RBF

##### Import input data
# x_val = pd.read_csv("code_testing\\x_training.csv", header=None).to_numpy() ## Reading a csv file with all parameters in each row
# # x_val = pd.read_csv("filename.csv", header=None, delim_whitespace=True).to_numpy() ## reading space delimited file

# x_val = x_val.astype('float64')
# # print(x_val)

# ##### import known outputs
# y_val = pd.read_csv("code_testing\\y_training.csv", header=None).to_numpy() ## Reading a csv file with all parameters in each row
# # y_val = pd.read_csv("filename.csv", header=None, delim_whitespace=True).to_numpy() ## reading space delimited file

# y_val = y_val.astype('int')
# y_val = np.reshape(y_val, -1)
# print(y_val)

x_val = pd.read_csv(input_path+basin_name+'\\'+basin_name+'_training.csv', header=None).to_numpy() ## Reading a csv file with all parameters in each row
# x_val = pd.read_csv("filename.csv", header=None, delim_whitespace=True).to_numpy() ## reading space delimited file

x_val = x_val[:,3:len(x_val[1,:])]
x_val = x_val.astype('float64')
# print(x_val)

##### import known outputs
y_val = pd.read_csv(input_path+basin_name+'\\'+basin_name+'_training.csv', header=None).to_numpy() ## Reading a csv file with all parameters in each row
# y_val = pd.read_csv("filename.csv", header=None, delim_whitespace=True).to_numpy() ## reading space delimited file

y_val = y_val[:,2].astype('int')
y_val = np.reshape(y_val, -1)
# print(y_val)


##### Fitting

### SVC
svc_clf = make_pipeline(StandardScaler(), SVC(C=5, kernel = 'rbf', tol=0.001, max_iter=-1, random_state=5))
svc_fit = svc_clf.fit(x_val, y_val)

score = svc_fit.score(x_val, y_val)
params = svc_fit.get_params()

y_model_cal = svc_fit.predict(x_val)
prec_cal = precision_score(y_val, y_model_cal, average='weighted')
recal_cal = recall_score(y_val, y_model_cal, average='weighted')

#from sklearn.metrics import PrecisionRecallDisplay
#display = PrecisionRecallDisplay.from_estimator(
#    svc_fit, x_val, y_val, name="rbf")
#display.ax_.set_title("2-class Precision-Recall curve")

#### SVC calculating some results
print("Calibration")
print("Cal Score: ",score)
print("Precision Score: ",prec_cal)
print("Recall Score: ",recal_cal)

print("")
print("Information Ratio")
for i in range(len(x_val[0])):
    x_temp = x_val[:,i].reshape(-1,1)
    igr_cal = mutual_info_classif(x_temp, y_val)
    ## , discrete_features=True
    # igr_cal = info_gain.info_gain_ratio(y_val, x_val)
    print(igr_cal, end = ', ')
print("")

alpha_cal = 0
correct_T_cal = 0
correct_F_cal = 0
beta_cal = 0
total_true_cal = 0
total_false_cal = 0
for i in range(len(y_val)):
    if(y_val[i] == 1 and y_model_cal[i] == 0):
        beta_cal += 1
        total_true_cal += 1
    elif(y_val[i] == 0 and y_model_cal[i] == 1):
        alpha_cal += 1
        total_false_cal += 1
    elif(y_val[i] ==0 and y_model_cal[i] == 0):
        correct_F_cal += 1
        total_false_cal += 1
    elif(y_val[i] ==1 and y_model_cal[i] == 1):
        correct_T_cal += 1
        total_true_cal += 1

print("True-True: ", correct_T_cal)
print("True-False: ", beta_cal)
print("False-True: ", alpha_cal)
print("False-Flase: ", correct_F_cal)
print("")
print("Total actual True: ", total_true_cal)
print("Total actual False: ", total_false_cal)
print("End Calibration")



# #### SVR
# svr_fit = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=1e-12, kernel = 'rbf', tol=0.001, max_iter=-1, random_state=5))
# svr_fit.fit(x, y)

# y_fit = np.zeros((len(x),1))

# # score = svr_fit.score(x, y)
# params = svr_fit.get_params()

# for i in len(x):
#     if(svr_fit.predict(x[i]) >= 0.5):
#         y_fit[i] = 1

### Visualize results
# plot_decision_regions(X=x_val, 
#                       y=y_val,
#                       clf=svc_clf, 
#                       legend=2)

# # Update plot object with X/Y axis labels and Figure Title
# plt.xlabel('Longitude', size=14)
# plt.ylabel('Latitude', size=14)
# plt.title('SVM Decision Region Boundary', size=16)
# plt.show()

###### Validation
##### Import validation data
x_valid = pd.read_csv(input_path+basin_name+'\\'+basin_name+'_validation.csv', header=None).to_numpy() ## Reading a csv file with all parameters in each row
x_valid = x_valid[:,3:len(x_valid[1,:])]
x_valid = x_valid.astype('float64')
# print(x_valid.shape)
y_valid = pd.read_csv(input_path+basin_name+'\\'+basin_name+'_validation.csv', header=None).to_numpy() ## Reading a csv file with all parameters in each row
y_valid = y_valid[:,2].astype('int')
y_valid = np.reshape(y_valid, -1)

y_model_val = svc_fit.predict(x_valid)

###### Calculate results
score = svc_fit.score(x_valid, y_valid)
prec_val = precision_score(y_valid, y_model_val, average='weighted')
recal_val = recall_score(y_valid, y_model_val, average='weighted')
print("")
print("Validation")
print("Val Score: ",score)
print("Precision Score: ",prec_val)
print("Recall Score: ",recal_val)
print("")
print("Information Ratio")
for i in range(len(x_val[0])):
    x_temp = x_valid[:,i].reshape(-1,1)
    igr_cal = mutual_info_classif(x_temp, y_valid)
    ## , discrete_features=True
    # igr_cal = info_gain.info_gain_ratio(y_val, x_val)
    print(igr_cal, end = ', ')
print("")

alpha_val = 0
correct_T_val = 0
correct_F_val = 0
beta_val = 0
total_true_val = 0
total_false_val = 0
for i in range(len(y_valid)):
    if(y_valid[i] == 1 and y_model_val[i] == 0):
        beta_val += 1
        total_true_val += 1
    elif(y_valid[i] == 0 and y_model_val[i] == 1):
        alpha_val += 1
        total_false_val += 1
    elif(y_valid[i] ==0 and y_model_val[i] == 0):
        correct_F_val += 1
        total_false_val += 1
    elif(y_valid[i] ==1 and y_model_val[i] == 1):
        correct_T_val += 1
        total_true_val += 1

print("True-True: ", correct_T_val)
print("True-False: ", beta_val)
print("False-True: ", alpha_val)
print("False-False: ", correct_F_val)
print("")
print("Total actual True: ", total_true_val)
print("Total actual False: ", total_false_val)
print("End Validation")

#### Checking for all points in the grrid
path_save = 'C:\\Users\\Dell\\Ver_0_hari1\\code\\Results\\'+basin_name+'\\SVM\\'
path_grid = 'C:\\Users\\Dell\\Ver_0_hari1\\code\\Processed_data\\'+basin_name+'\\grids\\'

grids_input_file_0 = pd.read_csv(path_grid+"grid_input.csv", header=None).to_numpy() ## Reading a csv file with all parameters in each row
grids_input_file = grids_input_file_0[:,2:len(grids_input_file_0[1,:])]
# print(grids_input_file)
grids_input_file = grids_input_file.astype('float64')

y_model_grid = svc_fit.predict(grids_input_file)

grid_outout = np.full((grids_input_file.shape[0],3), None)
#print(grid_outout.shape)

grid_outout[:,0] = grids_input_file_0[:,0]
grid_outout[:,1] = grids_input_file_0[:,1]
#print(grids_input_file_0[:,0])
grid_outout[:,2] = y_model_grid

np.savetxt(path_save+'grid_results.csv', grid_outout, delimiter=',')


###### Create Output netCDF
input_file_path = path_save
path_save = 'C:\\Users\\Dell\\Ver_0_hari1\\code\\Results\\'+basin_name+'\\SVM\\'

grids_input_file = pd.read_csv(input_file_path+"grid_results.csv", header=None).to_numpy()
grids_input_file = grids_input_file[:,:]
grids_input_file = grids_input_file.astype('float64')

pathf = 'C:\\Users\\Dell\\Ver_0_hari1\\code\\Data\\'+str(basin_name)+'\\input_layers\\netCDF'
    
read_data=Dataset(pathf+'\\Dist_Stream.nc','r')
var=list(read_data.variables.keys())

read_lat = read_data.variables['lat'][:]
read_lon = read_data.variables['lon'][:]

lonrect = read_lon[:]
latrect = read_lat[:]

l = np.full((len(latrect), len(lonrect)), None)
l = extract_data(latrect, lonrect, grids_input_file, l)

ds=Dataset(path_save+'Flood_Susceptibility.nc', mode="w", format='NETCDF4')
# some file-level meta-data attributes:
ds.Conventions = "CF-1.6" 
ds.title = 'Flood Susceptibility'


nlat=ds.createDimension('lat', len(latrect))
nlon=ds.createDimension('lon', len(lonrect))


longitude = ds.createVariable("lon","f8",("lon",))
latitude = ds.createVariable("lat","f8",("lat",))
longitude.units = "degrees_east"
latitude.units = "degrees_north"

humr = ds.createVariable("Flood_Susceptibility","f8",("lat","lon",),
                            chunksizes=(len(latrect),len(lonrect)), zlib=True, complevel=9)

humr.description = "1: Flood Susceptible, 0: Non-Susceptible to Floods"

longitude[:] = lonrect.reshape((len(lonrect),1))
latitude[:] = latrect.reshape((len(latrect),1))
humr[:] = l

ds.close()