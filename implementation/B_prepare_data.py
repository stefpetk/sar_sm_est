#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# MODULES NEEDED FOR THE PROGRAM #
import pandas as pd
import numpy as np
import gdal
import math


class model_variables:
    def __init__(self, date_range, image_list, s0_timeseries, sm_datafile, clay_pct, sand_pct, freq_GHZ):
        """

        Args:
            date_range: a list or numpy array which contains the sorted acquisition dates of the S1 and ISMN data (if
            two sets of .csv files are given for each polarization the variable should be a list or tuple of two lists)

            image_list: a list with the paths to the S1 GRD images

            sm_datafile: a list with the .csv files as pandas dataframes associated with the ISMN measurements
            (soil moisture, soil temperature, air temperature, precipitation) or the soil moisture values only for each
            station (if two sets of .csv files are given for each polarization the variable should be a list or tuple of
            two lists)

            s0_timeseries: a list or tuple of two .csv (one for the HV and one for the VV polarization) files which
            contain the backscatter coefficient timeseries for each station

            clay_pct: the percentage of clay in the soil where the ISMN station is located

            sand_pct: the percentage of sand in the soil where the ISMN station is located

            freq_GHZ: the frequency of the S1 radar signal

        """
        self.date_range = date_range
        self.sm_datafile = sm_datafile
        self.image_list = image_list
        self.s0_timeseries = s0_timeseries
        self.clay_pct = clay_pct
        self.sand_pct = sand_pct
        self.freq_GHZ = freq_GHZ

    def constr_in_par_df(self):

        """
        Function which creates a dataframe with the columns:
        's0_SAR_HV (dB)' --> (backscatter coefficient of the S1 radar signal in the HV polarization computed as the mean
        value of the backscatter coefficients of all the GRD image pixels which surround the ISMN station),
        's0_SAR_VV (dB)' --> (backscatter coefficient of the S1 radar signal in the VV polarization, computed in the
        same way as above)
        'Inc_Ang (rad)' --> (incidence angle of the S1 radar signal, computed in the same way as above)
        'SM (%)' --> (volumetric soil moisture measured for the ISMN station)
        'S_Temp_(Celsius)' --> (soil temperature measured in Celsius for the ISMN station)
        'Air_Temp_(Celsius)' --> (air temperature measured in Celsius for the ISMN station)
        'Precip_(mm)' --> (precipitation measured in millimeters for the ISMN station)
        'Below_Freezing_Temp' --> a boolean which indicates if the soil moisture or temperature measurement was made in
        freezing conditions

        Returns: all_vars_df, a pd.dataframe with the before mentioned columns

        """
        if self.image_list is not None:
            all_vars_df = pd.DataFrame(columns=['s0_SAR_HV_(dB)', 's0_SAR_VV_(dB)', 'Inc_Ang_(rad)',
                                                'SM_(m^3/m^3)', 'Soil_Temp_(Celsius)', 'Air_Temp_(Celsius)',
                                                'Precip_(mm)', 'Below_Freezing_Temp'],
                                       index=self.date_range)
            # initialization of the dataframe for the storing of #
            # the SAR backscatter coefficient values together with an index which contains the dates

            sigma0_mean_HV = []
            sigma0_mean_VV = []
            inc_ang_mean = []  # initialization of 3 lists where the mean value of the SAR backscatter coefficients and
            # the mean value of the local incidence angles for each pixel of the image will be saved in order to be
            # stored in the final dataframe

            for image, i in zip(self.image_list, range(len(self.image_list))):
                sar_cl_img = gdal.Open(image, gdal.GA_ReadOnly)
                # assert sar_cl_img.ReadAsArray().shape[0] == 3

                sar_cl_array_VV = sar_cl_img.GetRasterBand(1).ReadAsArray()
                if sar_cl_img.ReadAsArray().shape[0] == 3:
                    sar_cl_array_HV = sar_cl_img.GetRasterBand(2).ReadAsArray()
                    inc_ang_array = sar_cl_img.GetRasterBand(3).ReadAsArray()
                else:
                    sar_cl_array_HV = np.nan
                    inc_ang_array = sar_cl_img.GetRasterBand(2).ReadAsArray()
                # read the HV and VV bands of the SAR images as arrays

                if sar_cl_img.ReadAsArray().shape[0] == 3:
                    sigma0_mean_HV.append(np.nanmean(sar_cl_array_HV[:]))
                else:
                    sigma0_mean_HV.append(np.nan)
                sigma0_mean_VV.append(np.nanmean(sar_cl_array_VV[:]))
                inc_ang_mean.append(np.nanmean(inc_ang_array[:]))  # append the 2 lists with the HV and VV #
                # polarized backscatter coefficients for each date

                if sigma0_mean_HV[i] >= 0:
                    sigma0_mean_HV[i] = sigma0_mean_HV[i] == np.nan
                elif sigma0_mean_VV[i] >= 0:
                    sigma0_mean_VV[i] = sigma0_mean_VV[i] == np.nan
                else:
                    pass

            all_vars_df['s0_SAR_HV_(dB)'] = sigma0_mean_HV
            all_vars_df['s0_SAR_VV_(dB)'] = sigma0_mean_VV
            # write the list values to the dataframe

            for g in range(len(all_vars_df['Inc_Ang_(rad)'])):
                all_vars_df['Inc_Ang_(rad)'] = (inc_ang_mean[g] * math.pi) / 180.  # write the lists in the #
                # radar backscatter and incidence angle values dataframe

            for data_file in self.sm_datafile:
                if data_file.columns[2] == 'SM_(m^3/m^3)':
                    all_vars_df['SM_(m^3/m^3)'] = data_file['SM_(m^3/m^3)'].to_list()
                elif data_file.columns[2] == 'Soil_Temp_(Celsius)':
                    all_vars_df['Soil_Temp_(Celsius)'] = data_file['Soil_Temp_(Celsius)'].to_list()
                elif data_file.columns[2] == 'Air_Temp_(Celsius)':
                    all_vars_df['Air_Temp_(Celsius)'] = data_file['Air_Temp_(Celsius)'].to_list()
                elif data_file.columns[2] == 'Precip_(mm)':
                    all_vars_df['Precip_(mm)'] = data_file['Precip_(mm)'].to_list()
                else:
                    pass

                if data_file.columns[-1] == 'Below_Freezing_Temp':
                    all_vars_df['Below_Freezing_Temp'] = data_file['Below_Freezing_Temp'].to_list()
                    # fill the associated lists with all their elements

            all_vars_df.dropna(how='all', axis=1, inplace=True)
            # drop the columns of the initial parameters dataframe that contain null values

            return all_vars_df

        elif self.s0_timeseries is not None:
            all_vars_df_hv = pd.DataFrame(
                columns=['s0_SAR_HV_(dB)', 'Inc_Ang_(rad)', 'SM_(m^3/m^3)', 'Soil_Temp_(Celsius)',
                         'Air_Temp_(Celsius)', 'Precip_(mm)', 'Below_Freezing_Temp'],
                index=self.date_range[0])
            all_vars_df_vv = pd.DataFrame(
                columns=['s0_SAR_VV_(dB)', 'Inc_Ang_(rad)', 'SM_(m^3/m^3)', 'Soil_Temp_(Celsius)',
                         'Air_Temp_(Celsius)', 'Precip_(mm)', 'Below_Freezing_Temp'],
                index=self.date_range[1])
            # initialization of the dataframe for the storing of the SAR backscatter coefficient values for the VV and
            # VH polarizations together with an index which contains the dates

            all_vars_df_hv['s0_SAR_HV_(dB)'] = self.s0_timeseries[0]['sigma0_dB'].to_list()
            all_vars_df_hv['Inc_Ang_(rad)'] = self.s0_timeseries[0]['Mean_Incident_Angle_(rad)'].to_list()

            all_vars_df_vv['s0_SAR_VV_(dB)'] = self.s0_timeseries[1]['sigma0_dB'].to_list()
            all_vars_df_vv['Inc_Ang_(rad)'] = self.s0_timeseries[1]['Mean_Incident_Angle_(rad)'].to_list()
            # assign the values of the columns of the backscatter coefficient for the VH and VV polarization as well as
            # the incident angle

            all_vars_df_list = [all_vars_df_hv, all_vars_df_vv]

            for dataset, i in zip(self.sm_datafile, range(len(self.sm_datafile))):
                for data_file in dataset:
                    if data_file.columns[1] == 'SM_(m^3/m^3)':
                        all_vars_df_list[i]['SM_(m^3/m^3)'] = data_file['SM_(m^3/m^3)'].to_list()
                        all_vars_df_list[i]['Below_Freezing_Temp'] = data_file['Below_Freezing_Temp'].to_list()
                    elif data_file.columns[1] == 'Soil_Temp_(Celsius)':
                        all_vars_df_list[i]['Soil_Temp_(Celsius)'] = data_file['Soil_Temp_(Celsius)'].to_list()
                    elif data_file.columns[1] == 'Air_Temp_(Celsius)':
                        all_vars_df_list[i]['Air_Temp_(Celsius)'] = data_file['Air_Temp_(Celsius)'].to_list()
                    elif data_file.columns[1] == 'Precip_(mm)':
                        all_vars_df_list[i]['Precip_(mm)'] = data_file['Precip_(mm)'].to_list()
                    else:
                        pass
                    # fill the associated lists with all their elements

            return all_vars_df_list
        else:
            pass

    def e_estimation(self, ismn_data=True):
        """
        Function which estimates the real part of the complex dielectric constant for each date for which S1 Radar
        signal backscatter coefficients and mean volumetric soil moisture values are available according to the
        empirical models proposed by Topp et al. 1980 and Hallikainen et al. 1985. It also computes the soil moisture
        from the real part of the complex dielectric constant for each date and dielectric model.

        Returns: e_real_df, a pd.dataframe which contains a column with the real  and imaginary parts of the complex
        dielectric constant computed with the Hallikainen model

        """
        if self.image_list is not None or not ismn_data:
            for data_file in self.sm_datafile:
                if ismn_data:
                    data_file_cond = data_file.columns[2]
                else:
                    data_file_cond = str(type(data_file))

                if data_file_cond == "SM_(m^3/m^3)" or data_file_cond == f"<class 'pandas.core.series.Series'>":
                    mv_dec = data_file['SM_(m^3/m^3)'].to_numpy() if ismn_data else data_file.to_numpy()
                    # insertion of the pandas series with the mean values of #
                    # the measured soil moisture values for each date and conversion of the elements to decimal units

                    st_var = np.array([1, float(self.sand_pct), float(self.clay_pct)])
                    F = np.array([1.4, 4, 6, 8, 10, 12, 14, 16, 18])
                    E_r = np.zeros((len(self.date_range), len(F)))
                    E_i = np.zeros((len(self.date_range), len(F)))
                    e_r_hallikainen = np.zeros((len(self.date_range),))
                    e_i_hallikainen = np.zeros((len(self.date_range),))
                    # initialization of the E_r and E_i matrices from where a temporary
                    # value for the real and imaginary parts of the dielectric constant per Hallikainen et al. is
                    # computed

                    a_r = np.array([[+2.862, -0.012, +0.001],
                                    [+2.927, -0.012, -0.001],
                                    [+1.993, +0.002, +0.015],
                                    [+1.997, +0.002, +0.018],
                                    [+2.502, -0.003, -0.003],
                                    [+2.200, -0.001, +0.012],
                                    [+2.301, +0.001, +0.009],
                                    [+2.237, +0.002, +0.009],
                                    [+1.912, +0.007, +0.021]])

                    a_i = np.array([[0.356, -0.003, -0.008],
                                    [0.004, 0.001, 0.002],
                                    [-0.123, 0.002, 0.003],
                                    [-0.201, 0.003, 0.003],
                                    [-0.070, 0.000, 0.001],
                                    [-0.142, 0.001, 0.003],
                                    [-0.096, 0.001, 0.002],
                                    [-0.027, -0.001, 0.003],
                                    [-0.071, 0.000, 0.003]])

                    b_r = np.array([[+3.803, +0.462, -0.341],
                                    [+5.505, +0.371, +0.062],
                                    [+38.086, -0.176, -0.633],
                                    [+25.579, -0.017, -0.412],
                                    [+10.101, +0.221, -0.004],
                                    [+26.473, +0.013, -0.523],
                                    [+17.918, +0.084, -0.282],
                                    [+15.505, +0.076, -0.217],
                                    [+29.123, -0.190, -0.545]])

                    b_i = np.array([[5.507, 0.044, -0.002],
                                    [0.951, 0.005, -0.010],
                                    [7.502, -0.058, -0.116],
                                    [11.266, -0.085, -0.155],
                                    [6.620, 0.015, -0.081],
                                    [11.868, -0.059, -0.225],
                                    [8.583, -0.005, -0.153],
                                    [6.179, 0.074, -0.086],
                                    [6.938, 0.029, -0.128]])

                    c_r = np.array([[+119.006, -0.500, +0.633],
                                    [+114.826, -0.389, -0.547],
                                    [+10.720, +1.256, +1.522],
                                    [+39.793, +0.723, +0.941],
                                    [+77.482, -0.061, -0.135],
                                    [+34.333, +0.284, +1.062],
                                    [+50.149, +0.012, +0.387],
                                    [+48.260, +0.168, +0.289],
                                    [+6.960, +0.822, +1.195]])

                    c_i = np.array([[17.753, -0.313, 0.206],
                                    [16.759, 0.192, 0.290],
                                    [2.942, 0.452, 0.543],
                                    [0.194, 0.584, 0.581],
                                    [21.578, 0.293, 0.332],
                                    [7.817, 0.570, 0.801],
                                    [28.707, 0.297, 0.357],
                                    [34.126, 0.143, 0.206],
                                    [29.945, 0.275, 0.377]])
                    # matrices with the coefficients needed for the computation of the temporary electrical permittivity
                    # to the empirical model proposed by Hallikainen et al. 1985 #

                    for i in range(len(mv_dec)):
                        for j in range(len(F)):
                            E_r[i][j] = (a_r[j] @ st_var) + (b_r[j] @ st_var) * mv_dec[i] + (c_r[j] @ st_var) * \
                                        mv_dec[i] ** 2
                            E_i[i][j] = ((a_i[j] @ st_var) + (b_i[j] @ st_var) * mv_dec[i] + (c_i[j] @ st_var) *
                                         mv_dec[i] ** 2)

                    for k in range(len(E_r[:])):
                        e_r_hallikainen[k] = np.interp(self.freq_GHZ, F, E_r[k][:])
                        e_i_hallikainen[k] = np.interp(self.freq_GHZ, F, E_i[k][:])
                        # estimation of the real part of the electrical permittivity complex function by linear
                        # interpolation according to the central frequency of the Sentinel-1 constellation #

                    e_real_df = pd.DataFrame(data={'e_real_(Hallikainen)': e_r_hallikainen,
                                                   'e_imag_(Hallikainen)': e_i_hallikainen},
                                             index=self.date_range)

                    return e_real_df

        elif self.s0_timeseries is not None:
            e_real_df_list = []
            for dataset, i in zip(self.sm_datafile, range(len(self.sm_datafile))):
                for data_file in dataset:
                    if data_file.columns[1] == 'SM_(m^3/m^3)':
                        mv_dec = data_file['SM_(m^3/m^3)'].to_numpy()
                        # insertion of the pandas series with the mean values of #
                        # the measured soil moisture values for each date and conversion of the elements to decimal
                        # units

                        st_var = np.array([1, float(self.sand_pct), float(self.clay_pct)])
                        F = np.array([1.4, 4, 6, 8, 10, 12, 14, 16, 18])
                        E_r = np.zeros((len(self.date_range[i]), len(F)))
                        E_i = np.zeros((len(self.date_range[i]), len(F)))
                        e_r_hallikainen = np.zeros((len(self.date_range[i]),))
                        e_i_hallikainen = np.zeros((len(self.date_range[i]),))  # initialization of the E_r matrix from
                        # where the real and imaginary value of the dielectric constant per Hallikainen et al. is
                        # computed

                        # matrices with the coefficients needed for the computation of the dielectric constant to the
                        # empirical model proposed by Hallikainen et al. 1985 #

                        a_r = np.array([[+2.862, -0.012, +0.001],
                                        [+2.927, -0.012, -0.001],
                                        [+1.993, +0.002, +0.015],
                                        [+1.997, +0.002, +0.018],
                                        [+2.502, -0.003, -0.003],
                                        [+2.200, -0.001, +0.012],
                                        [+2.301, +0.001, +0.009],
                                        [+2.237, +0.002, +0.009],
                                        [+1.912, +0.007, +0.021]])

                        a_i = np.array([[0.356, -0.003, -0.008],
                                        [0.004, 0.001, 0.002],
                                        [-0.123, 0.002, 0.003],
                                        [-0.201, 0.003, 0.003],
                                        [-0.070, 0.000, 0.001],
                                        [-0.142, 0.001, 0.003],
                                        [-0.096, 0.001, 0.002],
                                        [-0.027, -0.001, 0.003],
                                        [-0.071, 0.000, 0.003]])

                        b_r = np.array([[+3.803, +0.462, -0.341],
                                        [+5.505, +0.371, +0.062],
                                        [+38.086, -0.176, -0.633],
                                        [+25.579, -0.017, -0.412],
                                        [+10.101, +0.221, -0.004],
                                        [+26.473, +0.013, -0.523],
                                        [+17.918, +0.084, -0.282],
                                        [+15.505, +0.076, -0.217],
                                        [+29.123, -0.190, -0.545]])

                        b_i = np.array([[5.507, 0.044, -0.002],
                                        [0.951, 0.005, -0.010],
                                        [7.502, -0.058, -0.116],
                                        [11.266, -0.085, -0.155],
                                        [6.620, 0.015, -0.081],
                                        [11.868, -0.059, -0.225],
                                        [8.583, -0.005, -0.153],
                                        [6.179, 0.074, -0.086],
                                        [6.938, 0.029, -0.128]])

                        c_r = np.array([[+119.006, -0.500, +0.633],
                                        [+114.826, -0.389, -0.547],
                                        [+10.720, +1.256, +1.522],
                                        [+39.793, +0.723, +0.941],
                                        [+77.482, -0.061, -0.135],
                                        [+34.333, +0.284, +1.062],
                                        [+50.149, +0.012, +0.387],
                                        [+48.260, +0.168, +0.289],
                                        [+6.960, +0.822, +1.195]])

                        c_i = np.array([[17.753, -0.313, 0.206],
                                        [16.759, 0.192, 0.290],
                                        [2.942, 0.452, 0.543],
                                        [0.194, 0.584, 0.581],
                                        [21.578, 0.293, 0.332],
                                        [7.817, 0.570, 0.801],
                                        [28.707, 0.297, 0.357],
                                        [34.126, 0.143, 0.206],
                                        [29.945, 0.275, 0.377]])

                        for j in range(len(mv_dec)):
                            for k in range(len(F)):
                                E_r[j][k] = (a_r[k] @ st_var) + (b_r[k] @ st_var) * mv_dec[j] + (c_r[k] @ st_var) * \
                                            mv_dec[j] ** 2
                                E_i[j][k] = (a_i[k] @ st_var) + (b_i[k] @ st_var) * mv_dec[j] + (c_i[k] @ st_var) * \
                                            mv_dec[j] ** 2

                        for m in range(len(E_r[:])):
                            e_r_hallikainen[m] = np.interp(self.freq_GHZ, F, E_r[m][:])
                            e_i_hallikainen[m] = np.interp(self.freq_GHZ, F, E_i[m][:])
                            # estimation of the real and imaginary part of the dielectric constant complex function by
                            # linear interpolation according to the central frequency of the Sentinel-1 constellation

                        e_real_df = pd.DataFrame(
                            data={'e_real_(Hallikainen)': e_r_hallikainen,
                                  'e_imag_(Hallikainen)': e_i_hallikainen},
                            index=self.date_range[i])
                        e_real_df_list.append(e_real_df)
            return e_real_df_list
        else:
            pass
