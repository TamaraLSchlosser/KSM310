# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 15:36:52 2025

@author: tamaras9
"""
from IPython import get_ipython
# Clear all variables in Spyder's Variable Explorer
get_ipython().run_line_magic('reset', '-f')

import matplotlib.pyplot as plt
import numpy as np
import netCDF4 as nc
from datetime import datetime
from statsmodels.tsa.ar_model import AutoReg
from scipy.ndimage import uniform_filter1d,gaussian_filter1d

#%% Read SST and pick out locations of interest
# Open the netCDF file and read the SST data
file_path = "sst.mnmean.nc"  # Uploaded into Data folder in the File Browser window
dataset = nc.Dataset(file_path)

# Extract latitude, longitude, and SST variables
lats = np.array(dataset.variables['lat'][:])
lons = np.array(dataset.variables['lon'][:])
sst = np.array(dataset.variables['sst'][:]) # time, lat, lon

time = dataset.variables['time'][:] # days since 1800-1-1 00:00:00
time_unit = dataset.variables["time"].getncattr('units') # first read the 'units' attributes from the variable time
time=np.array(nc.num2date(time, time_unit))#
time = np.array([t.strftime("%Y-%m-%d") for t in time], dtype="datetime64")

# Debug: Check the converted time
print("Converted time example:", time[:5])

# Close the dataset
dataset.close()

# function to find closest grid cell
def find_index(x,y):
    lat_idx = np.abs(lats - y).argmin()  # Find index of nearest latitude
    lon_idx = np.abs(lons - x).argmin()  # Find index of nearest longitude
    return lon_idx, lat_idx

# testing it works
thisLong, thisLat = find_index(16,5)
print([thisLong, thisLat])
print(lons[thisLong])
print(lats[thisLat])

# Define the locations of interest (latitude, longitude)
locations = [
    {"name": "Byron Bay", "lat": -28.7, "lon": 156},
    {"name": "Perth", "lat": -32, "lon": 115},
]
nloc=len(locations)

sst_values = np.full((nloc, len(time)), np.nan)
for ii, loc in enumerate(locations):  # Enumerate provides the index (i) and the location (loc)
    lon_idx, lat_idx = find_index(loc["lon"],loc["lat"])
    sst_values[ii, :] = np.squeeze(sst[:, lat_idx, lon_idx])  # Assign the time series for this location, and squeeze to reduce dimensions

# focus on last 30 years
tind=time>=np.datetime64("1994-01-01")
year=(time[tind]-np.min(time[tind])).astype('timedelta64[D]').astype('float32') / 365.25
sst_values=sst_values[:,tind]
time=time[tind]

# smooth or filter out seasonal variability
# smooth via a 12-month running mean
sst_smooth= uniform_filter1d(sst_values, size=12, axis=1,mode='nearest')
# also try low-pass filter.
sst_lowpass=gaussian_filter1d(sst_values, sigma=10, axis=1)

#%% for seminar
plt.figure(figsize=(16,6))
plt.plot(time, sst_values[0,:], label="raw",linewidth=2)
plt.plot(time, sst_smooth[0,:], label="smoothed",linewidth=4)
plt.plot(time, sst_lowpass[0,:], label="filter",linewidth=4)
plt.legend()
plt.xlabel('Year')
plt.ylabel('Sea Surface Temperature (°C)')
plt.tight_layout()
plt.xlim(time[0], time[-1])
plt.savefig("Byron_SST.jpg", dpi=600, quality=95, bbox_inches='tight')

# I like the filtered result so I'll use that

#%% Fitting Auto-Regressive (AR1) model to SST data
# generate the future 10 years time variable
year_diff_median = np.median(np.diff(year))
n10=int(np.round(10/year_diff_median))
time_diff_median = np.median(np.diff(time))
future10 = np.arange(time[-1]+time_diff_median, time[-1]+time_diff_median*(n10+1),time_diff_median)

# Fit the AR1 model and predict for the next n10 years
predictions1 = np.full((3,n10),np.nan)
ar1_coefficients = []
for ii in range(sst_values.shape[0]):
    model = AutoReg(sst_lowpass[ii,:], lags=1)
    model_fitted = model.fit()
    prediction = model_fitted.predict(start=len(time),end=len(time)+n10-1)  # Predict for the next 10 years
    predictions1[ii,:]=prediction
    ar1_coefficients.append(model_fitted.params[1])  # AR1 coefficient is the lag parameter

# Output AR1 coefficients
for i, coeff in enumerate(ar1_coefficients):
    print(f"Location {i+1}: AR1 Coefficient = {coeff}")


#%% Plot the SST values for the locations
# Define the x-axis limits (use datetime objects for time)
start_date = datetime(1992, 1, 1)
end_date = datetime(2022, 1, 1)

plt.figure(figsize=(8, 6))
for ii, loc in enumerate(locations):
    plt.plot(time, sst_values[ii, :], label=loc["name"])  # Plot SST for each location

# Set x-axis limits
plt.xlim(start_date, end_date)

plt.title("Sea Surface Temperature (SST) at Selected Locations")
plt.ylabel("SST (°C)")
plt.xlabel("year")

# Rotate date labels for better visibility
plt.gcf().autofmt_xdate()

#plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.legend(title="Locations",loc="upper left", fontsize=10)
plt.grid(True)
plt.tight_layout()
plt.show()

#%% tips

# other functions that might be helpful: 
# np.polyfit - can fit line/polygon to data, defining "order" of fitted line/curve
# np.percentile - find 95th percentile
