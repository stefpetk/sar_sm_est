# Implementation Pipeline
1) B_prepare_data.py: Creates a dataframe with Sentinel-1 backscatter coefficients, incidence angles, and ISMN measurements.
2) B1_group_backscatter_data.py: Renames Sentinel-1 GRD images based on acquisition dates and removes redundant images.
* Filters and groups Sentinel-2 NDVI/NDMI time series and ISMN data by user-defined thresholds.
* Uses the Hampel filter to remove outliers.
3) C1_backscatter_models_dataframes.py: Computes modeled backscatter coefficients using empirical (e.g., New Empirical Model) and physical (e.g., IEM) models.
4) C2_merge_backscatter_dfs.py: Merges modeled and SAR-acquired backscatter coefficient dataframes, selecting optimal models based on bias.
5) D1_Filter_Backscatter_dfs.py: Filters outliers in backscatter coefficient dataframes using statistical methods and generates comparison plots.
6) D2_Inversion.py: Performs soil moisture inversion using backscatter models and computes RMSE and R² metrics.   
