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
import xarray as xr
from datetime import datetime
from statsmodels.tsa.ar_model import AutoReg
from scipy.ndimage import uniform_filter1d,gaussian_filter1d

#%% Read SST and pick out locations of interest
# Open the netCDF file and read the SST data
file_path = "sst.mnmean.nc"  # Uploaded into Data folder in the File Browser window
dataset = xr.open_dataset(file_path,decode_times=True)

# Extract latitude, longitude, and SST variables
lats = dataset['lat'].values
lons = dataset['lon'].values
sst = dataset['sst'].values # time, lat, lon

time = dataset['time'].values

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
plt.figure(figsize=(8,3))
plt.plot(time, sst_values[0,:], label="raw",linewidth=2)
plt.plot(time, sst_smooth[0,:], label="smoothed",linewidth=4)
plt.plot(time, sst_lowpass[0,:], label="filter",linewidth=4)
plt.legend()
plt.xlabel('Year')
plt.ylabel('Sea Surface Temperature (°C)')
plt.tight_layout()
plt.xlim(time[0], time[-1])
plt.savefig("Byron_SST.jpg", dpi=600, bbox_inches='tight')

# I like the filtered result so I'll use that


#%% Plot the SST values for the locations
# Define the x-axis limits (use datetime objects for time)
start_date = datetime(1992, 1, 1)
end_date = datetime(2022, 1, 1)

plt.figure(figsize=(6, 4))
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

#%% Compute the long-term trend over last 30 years
trends = np.zeros(nloc)
fitted = np.full((nloc,len(year)),np.nan)
for ii in range(nloc):
    trend = np.polyfit(year, sst_lowpass[ii,:], 1)  # Linear trend (slope, intercept)
    trends[ii]=trend[0] # deg C/year
    fitted[ii,:]=np.polyval(trend, year)

# Output trend values
for ii, trends in enumerate(trends):
    print(f"Location {ii+1}: 30-year trend = {trends}")

# plot trend lines
plt.figure(figsize=(6, 4))
for ii, loc in enumerate(locations):
    plt.plot(time, sst_values[ii, :], label=loc["name"])  # Plot SST for each location
    plt.plot(time,fitted[ii,:],'--')


#%% thermal stress
thermal_stress_temp = np.percentile(sst_values, 95,axis=1) # SST threshold for thermal stress

plt.figure(figsize=(6,3))
for ii,loc in enumerate(locations):
    plt.plot(time, sst_values[ii,:], label=loc["name"])
    # now add a red cross for thermal stress events
    stress_ind=sst_values[ii,:]>thermal_stress_temp[ii]
    plt.scatter(time[stress_ind],sst_values[ii,stress_ind],marker='x',c='red',label="stress")

#%% Fitting Auto-Regressive (AR1) model to SST data
# generate the future 10 years time variable
year_diff_median = np.median(np.diff(year))
n10=int(np.round(10/year_diff_median))
time_diff_median = np.median(np.diff(time))
future10 = np.arange(time[-1]+time_diff_median, time[-1]+time_diff_median*(n10+1),time_diff_median)

# Fit the AR1 model and predict for the next n10 years
predictions1 = np.full((nloc,n10),np.nan)
ar1_coefficients = []

# Let's also save xi so we can see how the 'forcing' impacts our prediction
rng = np.random.default_rng()  # Optional: set seed for reproducibility seed=42
increase_noise=2 # feel free to change this if you want more 'oscillatory' results
xi=np.full((nloc,n10),np.nan) # our noise/forcing

for ii in range(sst_values.shape[0]):
    # fit easing least-squares
    model = AutoReg(sst_lowpass[ii,:], lags=1)
    model_fitted = model.fit()
    
    intercept = model_fitted.params[0]
    alpha = model_fitted.params[1]  # AR1 coefficient
    residual_std = np.std(model_fitted.resid)# residual standard deviation, used to scale noise/forcing/xi
    
    # Initialize prediction array
    pred = np.zeros(n10)
    # define our xi or forcing
    xi[ii,:]=residual_std * rng.normal(size=n10) * increase_noise
    # find first forecasted value
    pred[0] = alpha * sst_lowpass[ii, -1] + intercept + xi[ii,0]

    # forecast next 10 years
    for tt in range(1, n10):
        pred[tt] = alpha * pred[tt-1] + intercept + xi[ii,tt-1]

    predictions1[ii, :] = pred.copy()
    ar1_coefficients.append(model_fitted.params[1])  # AR1 coefficient is the lag parameter

# Output AR1 coefficients
for i, coeff in enumerate(ar1_coefficients):
    print(f"Location {i+1}: AR1 Coefficient = {coeff}")

# Plot the predictions
plt.figure(figsize=(8, 3))
for ii in range(sst_values.shape[0]):
    plt.plot(time, sst_lowpass[ii,:], label=f"Location {ii+1}")
    plt.plot(future10, predictions1[ii,:], label=f"Prediction AR1 {ii+1}", linestyle='--')
    

#%% What if we used a higher-order AR model? Like n10 order?
predictions10y = np.full((3,n10),np.nan)
alphas = np.full((3,n10),np.nan)

for ii in range(sst_values.shape[0]):
    model = AutoReg(sst_lowpass[ii,:], lags=n10)
    model_fitted = model.fit()
    prediction = model_fitted.predict(start=len(time),end=len(time)+n10-1)  # Predict for the next 10 years
    predictions10y[ii,:]=prediction
    alphas[ii,:] = model_fitted.params[1:] 

