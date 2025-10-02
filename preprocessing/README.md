# Preprocessing Pipeline
1)  A1_download_S1_data_script.js:
*  Downloads Sentinel-1 GRD images for a specified area and date range using Google Earth Engine.
*  Requires a GeoJSON file defining the area of interest.
*  Can be run in the Google Earth Engine Code Editor.
2) A2_manip_grd_name.py: Renames Sentinel-1 GRD images based on acquisition dates and removes redundant images.
3) A3_manip_station_files.py: Reformats ISMN data files from .stm to .csv and filters out unwanted measurement depths.
4) A4_fix_tseries.py: Aligns Sentinel-1 backscatter time series with ISMN data and converts gamma0 to sigma0 backscatter coefficients.
5) A5_filter_sm_ts_values: Filters ISMN soil moisture and temperature data to match Sentinel-1 acquisition dates and computes mean values.
6) A6_usa_align_dfs_for_B.py: Removes redundant rows from backscatter time series to align with ISMN data.
