from sentinelsat.sentinel import SentinelAPI, read_geojson, geojson_to_wkt
import glob
import os
import numpy as np
import pandas as pd


class fix_tseries:
    def __init__(self, geojson_path, username, password, Start_Time, End_Time, tseries_csv, folder_ending,
                 hv_vv_tseries_path):
        """
            Args:
                geojson_path: the path to the station folder containing geojson files
                username: the username for accessing the copernicus open access hub
                password: the corresponding password
                Start_Time: the start time for S1 GRD images searching as a string (YYYYMMDD)
                End_Time: the end time for S1 GRD images searching as a string (YYYYMMDD)
                tseries_csv: the path to the .csv file containing the timeseries with the mean value of the radar
                backscatter for each station
                folder_ending: a list containing the end characters of the folders where the geojson files and the final
                filtered backscatter coefficient timeseries are located for each station
                hv_vv_tseries_path: a tuple containing the paths to the HV and VV polarization gamma0 backscatter
                coefficients timeseries for each station (it can be also formatted so that it acts as a reference to
                multiple folders)
                """

        self.geojson_path = geojson_path
        self.username = username
        self.password = password
        self.Start_Time = Start_Time
        self.End_Time = End_Time
        self.tseries_csv = tseries_csv
        self.folder_ending = folder_ending
        self.hv_vv_tseries_path = hv_vv_tseries_path

    def tseries_times_func(self):
        """
        Returns: a modified dataframe containing the S1 GRD timeseries for each ISMN station
        """
        api = SentinelAPI(self.username, self.password, 'https://scihub.copernicus.eu/dhus')
        station_footprint = geojson_to_wkt(read_geojson(self.geojson_path))
        s1_grd_prods = api.query(area=station_footprint,
                                 date=(self.Start_Time, self.End_Time),
                                 platformname='Sentinel-1',
                                 producttype='GRD')
        s1_grd_prods_df = api.to_dataframe(s1_grd_prods)
        # generate a dataframe which contains the S1 GRD image filenames and metadata that are associated with a
        # geometry ( station_footprint) in a date range

        s1_grd_dates_times = s1_grd_prods_df[['title', 'relativeorbitnumber']].copy()
        s1_grd_dates = []
        s1_grd_times = []
        for i in range(len(s1_grd_dates_times['title'])):
            s1_grd_dates.append(
                s1_grd_dates_times['title'][i][17:21] + '-' + s1_grd_dates_times['title'][i][21:23] + '-' +
                s1_grd_dates_times['title'][i][23:25])
            s1_grd_times.append(
                s1_grd_dates_times['title'][i][26:28] + ':' + s1_grd_dates_times['title'][i][28:30] + ':' +
                s1_grd_dates_times['title'][i][30:32])

        s1_grd_dates.reverse()
        s1_grd_times.reverse()
        s1_grd_dates_times['date'] = s1_grd_dates
        s1_grd_dates_times['time'] = s1_grd_times
        s1_grd_dates_times['title'] = s1_grd_dates_times['title'][::-1].to_list()
        # create a new dataframe with the acquisition times and dates for each S1 GRD images daterange

        tseries_df = pd.read_csv(self.tseries_csv)
        tseries_df.drop_duplicates(subset='C0/date', inplace=True, ignore_index=True)
        s1_tseries_dlist = []
        for j in range(len(tseries_df['C0/date'])):
            s1_tseries_dlist.append(tseries_df['C0/date'][j][:10])
            # append the dates from the .csv files where the gamma0 timeseries are located

        s1_grd_dates_times.drop_duplicates(subset='date', inplace=True, ignore_index=True)
        s1_grd_dates_times = s1_grd_dates_times[s1_grd_dates_times['date'].isin(s1_tseries_dlist)]

        if len(s1_tseries_dlist) > len(s1_grd_dates_times['date']):
            for i in range(len(tseries_df['C0/date'])):
                tseries_df['C0/date'][i] = tseries_df['C0/date'][i][:10]
            tseries_df = tseries_df[tseries_df['C0/date'].isin(s1_grd_dates_times['date'].to_list())]
            tseries_df['C0/time'] = s1_grd_dates_times['time'].to_list()
            tseries_df['Rel_Orb_Num'] = s1_grd_dates_times['relativeorbitnumber'].to_list()
        else:
            tseries_df['C0/time'] = s1_grd_dates_times['time'].to_list()
            tseries_df['Rel_Orb_Num'] = s1_grd_dates_times['relativeorbitnumber'].to_list()
            for i in range(len(tseries_df['C0/date'])):
                tseries_df['C0/date'][i] = tseries_df['C0/date'][i][:10]

        return tseries_df

    def prod_sigma0_tseries(self):
        """
        Takes: the dataframes where the final filtered gamma0 backscatter coefficient timeseries are located
        Returns: the dataframe where the final filtered sigma0 backscatter coefficient timeseries
        """

        hv_pol_dfs = []
        vv_pol_dfs = []
        # initialize the lists where the backscatter coefficient dataframes will be stored for each ISMN station for the
        # HV and VV polarization respectively as well as the load the dataframe where the mean incident angle values for
        # each station and relative orbit number

        for n in self.folder_ending:
            hv_pol_dfs.append(pd.read_csv(glob.glob(self.hv_vv_tseries_path[0].format(n))[0]))
            vv_pol_dfs.append(pd.read_csv(glob.glob(self.hv_vv_tseries_path[1].format(n))[0]))
            # append the lists with the timeseries as dataframes for each station

        for i in range(len(hv_pol_dfs)):
            hv_pol_dfs[i]['gamma0_linear'] = 10 ** (hv_pol_dfs[i]['C0/mean'] / 10)
            hv_pol_dfs[i]['sigma0_linear'] = hv_pol_dfs[i]['gamma0_linear'] * \
                                             np.cos(hv_pol_dfs[i]['Mean_Incident_Angle_(rad)'])
            hv_pol_dfs[i]['sigma0_dB'] = 10 * np.log10(hv_pol_dfs[i]['sigma0_linear'])

            vv_pol_dfs[i]['gamma0_linear'] = 10 ** (vv_pol_dfs[i]['C0/mean'] / 10)
            vv_pol_dfs[i]['sigma0_linear'] = vv_pol_dfs[i]['gamma0_linear'] * \
                                             np.cos(vv_pol_dfs[i]['Mean_Incident_Angle_(rad)'])
            vv_pol_dfs[i]['sigma0_dB'] = 10 * np.log10(vv_pol_dfs[i]['sigma0_linear'])

        return hv_pol_dfs, vv_pol_dfs
