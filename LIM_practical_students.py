# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 16:17:39 2025

@author: tamaras9
"""

from IPython import get_ipython
# Clear all variables in Spyder's Variable Explorer
get_ipython().run_line_magic('reset', '-f')

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.signal import detrend
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.linalg import logm, pinv 

#%% Data prep: Open SST data, find annual mean, then detrend data
# Step 1: Open the netCDF file and read the SST data
file_path = "sst.mnmean.nc"  # Uploaded into Data folder in the File Browser window
dataset = xr.open_dataset(file_path,decode_times=True)

# Step 2: Extract latitude, longitude, and SST variables
lats = dataset['lat'].values
lons = dataset['lon'].values
sst = dataset['sst'].values # time, lat, lon
sst[sst<=-2]=np.NaN

time = dataset['time'].values

# Close the dataset
dataset.close()

# smooth via a 3-month running mean
sst_sm = uniform_filter1d(sst, size=3, axis=0, mode='nearest')

# Remove climatology (i.e., monthly average)
sst_anom=np.full_like(sst, np.nan)
for mm in range(12):
    climatology = np.nanmean(sst_sm[mm::12, :, :], axis=0)  # Compute climatology for month mm
    sst_anom[mm::12,:,:]=sst_sm[mm::12,:,:]-climatology

# now need to decrease run time by decreasing the size of our sst array
# use only last 70 years to match with Nino3 index
tind=time>=np.datetime64("1954-01-01")
sst_anom=sst_anom[tind,:,:]
sst=sst[tind,:,:]
time=time[tind]

# limit to +-60 lat
yind=np.abs(lats)<=60
lats=lats[yind]
sst_anom=sst_anom[:,yind,:]

# example code for plotting SST anomalies in one year
plt.figure(figsize=(8, 6))
plt.pcolor(lons,lats,sst_anom[0,:,:])
plt.colorbar(label="SST Anomaly - 1950")
plt.xlabel("longitude (°E)")
plt.ylabel("latitude (°N)")
plt.show()   

# we have now smoothed and demeaned the SST data

#%% Next vectorize each time step 
# I need the array to have dimensions [time x space] instead of [time x lat x lon]
tmp=sst_anom # [time x lats x lons]
# don't include grid cells with NaNs or consistently very small SST anomalies
tmp[np.isnan(tmp)]=0
count = np.sum(np.abs(tmp) < 1e-2, axis=0)
bad_ind=count>5*12 # if more than 5 years with "bad" data, then we want to omit
tmp=tmp[:,~bad_ind]

sst_vect = np.full((len(time),np.size(tmp, axis=1)), np.nan) # time x global cells
for tt in range(len(time)):
    # take a time slice
    tmp=sst_anom[tt,:,:]
    # remove nans/very small points
    tmp=tmp[~bad_ind]
    # assign
    sst_vect[tt,:]=tmp.copy()

plt.figure(figsize=(8, 6))
plt.pcolor(time[:12],range(sst_vect.shape[1]),sst_vect[:12,:].T)

# Rotate date labels for better visibility
plt.gcf().autofmt_xdate()

plt.colorbar(label="SST Anomaly (°C)")
plt.xlabel("1st 12 time steps")
plt.ylabel("global cells")
plt.show()

# next linearly detrend the data so that the mean is around 0 for all decades
sst_detrend=np.full_like(sst_vect, np.nan)
sst_trend=np.full_like(sst_vect, np.nan)
for ii in range(sst_vect.shape[1]):    
    # Apply detrending while ignoring NaNs
    sst_detrend[:,ii]= detrend(sst_vect[:,ii], type='linear')
    sst_trend[:,ii]=sst_vect[:,ii]-sst_detrend[:,ii]

# example code for plotting detrended SST anomalies in one grid cell
plt.figure(figsize=(8, 6))
plt.plot(time,sst_vect[:,0],color="blue",label="raw")
plt.plot(time,sst_detrend[:,0],color="red",label="detrended")
plt.plot(time,sst_trend[:,0],color="black",label="trend")
plt.hlines(0,time[0],time[-1],color='k',linestyles='--')
plt.legend()
plt.xlabel("date")
plt.ylabel("SST Anomaly (°C)")
plt.show()

#%% Apply latitude weights to SST anomalies for EOF analysis then normalise

# Prepare array with cos(latitude) values. sst_detrend will be multiplied by cos(latitude) before EOF calculation
coslat = np.cos(np.deg2rad(lats))
wgts = coslat[..., np.newaxis]
cossF = np.zeros_like(sst_anom)
# Broadcast wgts to all rows of cossF
cossF[:] = wgts

# Reshape array to (time, space)
coss_a = cossF.reshape(len(time),(len(lats) * len(lons)))
bad_ind_vect = bad_ind.reshape((len(lats) * len(lons)))
# Remove missing values from the design matrix.
coss_a = coss_a[:,~bad_ind_vect]

# Multiply by cosine of latitude
sst_anom_primed=sst_detrend*coss_a

# Normalise each field (this is needed when more than one variable is used)
# Compute the variance at each grid point, take the average, take the square root
vart = np.nanvar(sst_anom_primed,axis=0)
AA_sst=np.sqrt(np.nanmean(vart))

# Normalise
sst_anom_primed=sst_anom_primed/AA_sst

#%% Do EOF analysis for SST anomalies

# Perform EOF analysis: Compute EOFs, PCs, and explained variance
n_modes=5 # number of EOFs, adjustable

# Compute EOFs of SST
_, S1, C1 = np.linalg.svd(sst_anom_primed, full_matrices=False)

S1 = np.diag(S1)  # Convert singular values to a diagonal matrix
fve1 = np.round(np.diag(S1)**2/np.nansum(np.diag(S1)**2),2) # Fraction of variance explained
PCs = sst_anom_primed @ (C1.T) # Compute PCs by projecting  

# Retain the first n_modes PCs SST data on EOFs
PCs=PCs[:,:n_modes] # Chosen leading PCs, i.e., timeseries
EOFs=C1[:n_modes,:] # Chosen leading EOFs, i.e., spatial pattern
#(np.sum(fve1[:n_modes])) # Variance explained by n_modes, i.e., eigenvalues

scale=np.max(np.abs(EOFs[0,:]))

plt.figure(figsize=(8, 6))
plt.plot(time,PCs*scale)
plt.xlabel("date")
plt.ylabel("PCs (°C)")
plt.show()

#%% Retranslate vector [time x global cells] data into spatial map [time x lats x lons]

EOFs_world=np.full((n_modes,len(lats),len(lons)), np.nan)
clim=np.full((n_modes), np.nan) # colour limits when plotting
for ii in range(n_modes):
    tmp=np.full((len(lats),len(lons)), np.nan)
    tmp[~bad_ind]=EOFs[ii,:]
    EOFs_world[ii,:,:]=tmp
    # estimate good plotting colour limits
    clim[ii]=np.percentile(np.abs(EOFs[ii,:]),100)

#%% Plot all EOFs and PCs. 
# for plotting purposes, we will set the EOF amplitudes to ~=1 and modify PC accordingly

for mode in range(n_modes):
    fig, ax = plt.subplots(2, 1, figsize=(16, 8), gridspec_kw={'height_ratios': [2, 1]}, constrained_layout=True)

    # --- Top Plot: Spatial EOF ---
    levels = np.linspace(-1, 1, 53)
    eof_data = EOFs_world[mode, :, :]
    contour = ax[0].contourf(lons, lats, eof_data/scale, levels=levels, cmap='RdBu_r', vmin=-1, vmax=1)

    cbar = plt.colorbar(contour, ax=ax[0], orientation='vertical', fraction=0.05, pad=0.02)
    cbar.set_label(f"EOF{mode+1} Amplitude (unitless)", fontsize=14)

    ax[0].set_xlabel("Longitude (°E)", fontsize=12)
    ax[0].set_ylabel("Latitude (°N)", fontsize=12)
    ax[0].set_title(f"Spatial EOF-{mode+1}: {fve1[mode]*100}% variance", fontsize=12, fontweight="bold")
    ax[0].tick_params(axis='both', labelsize=11)

    # --- Bottom Plot: PC Time Series ---
    pc = PCs[:, mode]
    ax[1].plot(time, pc*scale, color='k', linewidth=2)
    ax[1].plot(time, np.zeros(len(time)), color='k', linestyle='--', linewidth=1)

    # Shading: Red for positive, blue for negative
    ax[1].fill_between(time, pc*scale, 0, where=(pc > 0), color='red', alpha=0.3)
    ax[1].fill_between(time, pc*scale, 0, where=(pc < 0), color='blue', alpha=0.3)

    ax[1].set_xlabel("date", fontsize=12)
    ax[1].set_ylabel("PC Amplitude (°C)", fontsize=12)
    ax[1].set_title(f"PC-{mode+1}", fontsize=12, fontweight="bold")
    ax[1].tick_params(axis='both', labelsize=11)
    ax[1].grid(True, linestyle="--", linewidth=0.5)
    ax[1].set_xlim(time[0], time[-1])

    # Save each figure
    plt.savefig(f"EOF{mode+1}_PC.jpg", dpi=600, bbox_inches='tight')
    #plt.close()


#%% LIM computation

# This option allows to compute a LIM operator over a subperiod, and make a prediction of a following period if desired
time_max=1980
time_LIM = np.datetime64(f"{time_max}-01-01")
ind_timel = time < time_LIM
timel = time[ind_timel]
neof = n_modes
ntm = len(timel)

# Define state vector to compute the LIM 
PC1 = PCs[ind_timel,:]
# subtract time mean of each PC; needed when using a sub-period
PC1 = PC1 - np.nanmean(PC1,axis=0)

# Calculate 0-lag covariance
C0 = ((PC1.T) @ PC1) / (PC1.shape[0]-1)

# Choose training lag (months)
tau0 = 1   # Training lag; needed to compute lag-covariance matrix

X0 = PC1[:-tau0,:] - np.nanmean(PC1[:-tau0,:],axis=0)
Xtau = PC1[tau0:,:] - np.nanmean(PC1[tau0:,:],axis=0)

# Calculate tau0-lag covariance, here tau0=1
Ctau = ((Xtau.T) @ X0)/ (X0.shape[0]-1)

# Compute LIM operators
G0 = Ctau @ pinv(C0)
L0 = logm(G0)/tau0 # our L linear operator
L0 = np.real(L0)

#%% Q1: Contrast modes with Nino3: https://psl.noaa.gov/data/timeseries/monthly/NINO3/
# Nino3: Central Tropical Pacific SST (5N-5S;150W-90W). Calculated from the NOAA ERSST V5.
central_yind=(lats>=-5) & (lats<=5)
central_xind=(lons>=210) & (lons<=270) # 150W=-150+360=210
# Use np.ix_ to create an indexer for the 2D region (latitude, longitude)
lat_ind, lon_ind = np.ix_(central_yind, central_xind)

# Load CSV while skipping the first row
file_path = "Nino3_index.csv"
Nino3 = pd.read_csv(file_path, skiprows=1,delim_whitespace=True, header=None)
Nino3 = Nino3.apply(pd.to_numeric, errors="coerce")  # Convert everything possible to numbers

# Rename columns (assuming the first column is Year, followed by 12 months)
Nino3.columns = ["Year", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# don't need last 3 rows
Nino3 = Nino3.drop(index=[78,79,80], axis=0)

# Convert "Year" column to integers
Nino3["Year"] = Nino3["Year"].astype(int)
# ERA5 data stops in 2023 and starts in 1954
Nino3 = Nino3[Nino3["Year"] <= 2023]
Nino3 = Nino3[Nino3["Year"] >= 1954]
Nino3 = Nino3.reset_index(drop=True)

# bad data was set to -99.99
Nino3.loc[:, Nino3.columns != "Year"] = Nino3.loc[:, Nino3.columns != "Year"].where(Nino3.loc[:, Nino3.columns != "Year"] >= -90, np.nan)

Nino3_index=Nino3.iloc[:,1:].to_numpy()
Nino3_index= Nino3_index.flatten()

time_Nino=time[0:len(Nino3_index)]

# plot Nino3 index
fig, (ax1,ax2) = plt.subplots(2, sharex=True, figsize=(8, 6))
ax1.plot(time_Nino,Nino3_index)
ax1.grid()
ax1.set(ylabel="Nino3 index")

# correlate Nino3 indices with mode amplitudes
spatial_map=np.full((len(time),len(lats[central_yind]),len(lons[central_xind])),np.nan)
CTP_ampl=np.full((len(time),n_modes),np.nan)
R=np.full(n_modes,np.nan)

# consider whether all modes are necessary in plot. Perhaps reduce to first few?
for ii in range(n_modes):
    sign=np.sign(np.mean(EOFs_world[ii,lat_ind, lon_ind]))
    Rtmp=np.corrcoef(PCs[0:len(Nino3_index),ii]*sign,Nino3_index)
    R[ii]=Rtmp[0,1]# correlation coefficient
    
    # now plot
    ax2.plot(time_Nino,PCs[0:len(Nino3_index),ii]*scale*sign,label=f"M{ii + 1}, R={R[ii]:.2f}")
    
plt.legend()
ax2.grid()
ax2.set_ylabel("PC (°C) * sign of EOF in CTP")

plt.savefig("Nino3_SST_modes.jpg", dpi=600, bbox_inches='tight')

print("Correlation coefficient with Nino3 Index:",np.round(R,2))

#%% Q2: SST anomalies vs Nino3
nloc=3
locations = [
    {"name": "Ningaloo", "lat": -22.6, "lon": 112},
    {"name": "GBR", "lat": -18, "lon": 150},
    {"name": "Galapogas", "lat": -0, "lon": -90+360},
]
# function to find closest grid cell
def find_index(x,y):
    lat_idx = np.abs(lats - y).argmin()  # Find index of nearest latitude
    lon_idx = np.abs(lons - x).argmin()  # Find index of nearest longitude
    return lon_idx, lat_idx

sst_values = np.full((nloc, len(time)), np.nan)
sst_anom_loc= np.full((nloc, len(time)), np.nan)
for ii, loc in enumerate(locations):  # Enumerate provides the index (i) and the location (loc)
    lon_idx, lat_idx = find_index(loc["lon"],loc["lat"])
    sst_values[ii, :] = np.squeeze(sst[:, lat_idx, lon_idx])  # Assign the time series for this location, and squeeze to reduce dimensions
    sst_anom_loc[ii, :] = np.squeeze(sst_anom[:, lat_idx, lon_idx])

fig, axes = plt.subplots(len(locations), sharex=True, figsize=(10, 6))

for ii, loc in enumerate(locations):
    ax1 = axes[ii]                        # Left y-axis (SST anomaly)
    ax2 = ax1.twinx()                     # Right y-axis (Nino3 index)
    
    # Plot SST anomaly
    ax1.plot(time, sst_anom_loc[ii, :], label=f"SST Anomaly - {loc['name']}", color='tab:blue')
    ax1.hlines(0,time[0],time[-1],color="black",linestyle="--")
    ax1.set_ylabel("SST Anom (°C)", color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ymin, ymax = ax1.get_ylim()
    ymax=np.max(np.abs([ymin,ymax]))
    ax1.set_ylim(-ymax,ymax)
    ax1.set_xlim(time[0],time[-1])
    
    # Plot Nino3 index
    ax2.plot(time_Nino, Nino3_index, label="Nino3", color='tab:red')
    ax2.set_ylabel("Nino3 Index", color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ymin, ymax = ax2.get_ylim()
    ymax=np.max(np.abs([ymin,ymax]))
    ax2.set_ylim(-ymax,ymax)
    ax2.set_xlim(time[0],time[-1])
    
    # Title for each subplot
    ax1.set_title(f"SST Anomaly at {loc['name']} vs Nino3")

# Shared x-label
plt.xlabel("time")
plt.tight_layout()
plt.savefig("SSTA_vs_Nino3_3locations.jpg", dpi=600, bbox_inches='tight')
plt.show()

#%% Q3: finding which modes contributed to the warmest SST anomalies

fig, axes = plt.subplots(len(locations), sharex=True, figsize=(10, 6))

# Check EOF*_PC.jpg then update accordingly. The dimensions are location x mode number
PC_sign=np.array([[1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1]])

for ii, loc in enumerate(locations):
    ax1 = axes[ii]                        # Left y-axis (SST anomaly)
    ax2 = ax1.twinx()                     # Right y-axis (Nino3 index)
    
    # Plot SST anomaly
    ax1.plot(time, sst_anom_loc[ii, :], label=f"SST Anomaly - {loc['name']}", color='tab:blue')
    ax1.hlines(0,time[0],time[-1],color="black",linestyle="--")
    ax1.set_ylabel("SST Anom (°C)", color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ymin, ymax = ax1.get_ylim()
    ymax=np.max(np.abs([ymin,ymax]))
    ax1.set_ylim(-ymax,ymax)
    ax1.set_xlim(time[0],time[-1])
    
    # Plot Nino3 index
    ax2.plot(time, PCs[:,0]*scale*PC_sign[ii,0], color='tab:red',label="mode-1")
    ax2.plot(time, PCs[:,1]*scale*PC_sign[ii,0], color='tab:green',label="mode-2")
    ax2.plot(time, PCs[:,2]*scale*PC_sign[ii,0], color='tab:purple',label="mode-3")
    ax2.set_ylabel("PCs (°C)", color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.legend(loc="upper left")
    ymin, ymax = ax2.get_ylim()
    ymax=np.max(np.abs([ymin,ymax]))
    ax2.set_ylim(-ymax,ymax)
    ax2.set_xlim(time[0],time[-1])
    
    # Title for each subplot
    ax1.set_title(f"SST Anomaly at {loc['name']} vs PC-1 ")

# Shared x-label
plt.xlabel("time")
plt.tight_layout()
plt.savefig("SSTA_vs_3PCs_3locations.jpg", dpi=600, bbox_inches='tight')
plt.show()

#%% Q4: LIM Eigenanalysis
# Eigenvalue analysis of L0
lambda_vals, lambda_vecs = np.linalg.eig(L0) # lambda_vals will be 𝜆=𝜎+𝑖𝜔
# Damping timescale: 
damping_t=-1/np.real(lambda_vals)
# Period (if oscillatory): 
period=2*np.pi/np.imag(lambda_vals)
# check for stationary modes and assign period of NaN
period[np.imag(lambda_vals) == 0] = np.nan

# Extract real and imaginary parts
real_parts = np.real(lambda_vals)# Eigenvalues of L0 (Real parts)
imag_parts = np.imag(lambda_vals)# Eigenvalues of L0 (Imaginary parts)

# Stability check
if np.any(real_parts > 0):
    print("WARNING: System may be unstable (positive eigenvalues detected).")
else:
    print("System is stable (all eigenvalues have non-positive real parts).")
    
print("Damping timescales (months):", np.round(damping_t, 2))
print("Period (months):", np.round(period, 2))
# Mode 1 & 2 are a conjugate pair (±31.4 months)
# Mode 3 & 4: another pair (±36.38 months)
# refer to Lou paper in seminar and their Table 2, where index 2 had 2 modes
