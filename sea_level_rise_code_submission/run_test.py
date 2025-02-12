import xarray as xr
import numpy as np


# Same as in https://github.com/iharp-institute/HDR-ML-Challenge-public/blob/main/codabench_files/ingestion_program/ingestion.py

def get_sla_array(files):
    """
    Read the NetCDF files and extract the 'sla' variable
    """
    # Read the NetCDF files
    ds = xr.open_dataset(files)

    # Extract the 'sla' variable
    X = ds['sla'].values

    # Close the dataset
    ds.close()

    return X


solution_dir = "./"

import sys
sys.path.insert(1, solution_dir)
from model import Model
m = Model()

nc_filename = "dt_ena_20021225_vDT2021.nc"
X = get_sla_array(nc_filename)

y_pred = m.predict(X)

print(nc_filename)
print(X.shape)
print(y_pred)