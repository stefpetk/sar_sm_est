# Soil Moisture Estimation from Sentinel-1 and ISMN Data

This repository contains scripts for estimating soil moisture using Sentinel-1 Ground Range Detected (GRD) images' backscattering indices and in-situ measurements from the International Soil Moisture Network (ISMN). The project includes preprocessing, implementation, and visualization scripts to process data, model backscatter coefficients, and estimate soil moisture.

# Project Overview

The goal of this project is to preprocess Sentinel-1 GRD images and ISMN in-situ measurements, compute modeled backscatter coefficients using empirical and physical models, and estimate volumetric soil moisture through inversion techniques. The pipeline includes data preparation, filtering, grouping, modeling, and visualization of results.

# Repository Structure
```plaintext
sar_sm_est/
├── scripts/
│   ├── preprocessing/
│   │   ├── A1_download_S1_data_script.js        # Downloads Sentinel-1 GRD images
│   │   ├── A2_manip_grd_name.py                 # Renames and filters GRD images
│   │   ├── A3_manip_station_files.py            # Reformats ISMN data to CSV
│   │   ├── A4_fix_tseries.py                    # Aligns and converts backscatter time series
│   │   ├── A5_filter_sm_ts_values.py            # Filters ISMN data by acquisition dates
│   │   └── A6_usa_align_dfs_for_B.py            # Aligns backscatter time series with ISMN data
│   ├── implementation/
│   │   ├── B_prepare_data.py                    # Prepares data for modeling
│   │   ├── B1_group_backscatter_data.py         # Groups data by NDVI/NDMI thresholds
│   │   ├── C1_backscatter_models_dataframes.py   # Computes modeled backscatter coefficients
│   │   ├── C2_merge_backscatter_dfs.py          # Merges backscatter dataframes
│   │   ├── D1_Filter_Backscatter_dfs.py         # Filters outliers in backscatter data
│   │   └── D2_Inversion.py                      # Performs soil moisture inversion
│   └── visualization/
│       └── PL_backscatter_models_plot_recreation.py  # Visualizes backscatter comparisons
```

# Important Note!
The above scripts were developed as part of my undergraduate thesis (relevant link: ), but I hope to improve them in the context of training machine learning models for soil moisture prediction using SAR images and ground measurements. However, if you have any suggestions for improvement, please let me know by creating an issue.
