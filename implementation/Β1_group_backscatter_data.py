import pandas as pd
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
from hampel import hampel  # https://pypi.org/project/hampel/


def group_st_sar_data(data_group, ndvi_tseries_path, ndmi_tseries_path, ndvi_mkdir, ndmi_mkdir, ndvi_mkdir_filt,
                      ndmi_mkdir_filt, in_par_dfs, sar_ims_or_tseries):
    """
    Function that filters (hampel filtering) the timeseries (1 year date range) of the Sentinel-2 acquired NDVI
    (Normalized Difference Vegetation Index) and NDMI (Normalized Difference Moisture Index) timeseries for each ISMN
    station and then groups the ISMN data and S1 backscatter coeffici

    Args:
        data_group: a string that indicates the folder where the data for each station is located
        ndvi_tseries_path: the path to the NDVI index timeseries .csv files for each ISMN station
        ndmi_tseries_path: the path to the NDMI index timeseries .csv files for each ISMN station
        ndvi_mkdir: make a folder for the initial NDVI timeseries for each ISMN station
        ndmi_mkdir: make a folder for the initial NDMI timeseries for each ISMN station
        ndvi_mkdir_filt: make a folder for the Hampel filtered NDVI timeseries for each ISMN station
        ndmi_mkdir_filt: make a folder for the Hampel filtered NDMI timeseries for each ISMN station
        in_par_dfs: initial parameters pandas dataframe (contains both insitu and satellite data) for each ISMN station
        sar_ims_or_tseries: a string which is used to identify if for a station GRD SAR Images have been used as

    Returns: a tuple containing dataframes with the grouped insitu and satellite for a set of ISMN stations

    """
    assert data_group in ['Africa', 'Australia', 'USA']
    assert sar_ims_or_tseries in ['SAR Images', 'Backscatter Coefficient Timeseries']

    ndvi_tseries_list = []
    ndmi_tseries_list = []
    ndvi_tseries_fnames = glob.glob(ndvi_tseries_path.format(data_group))
    ndmi_tseries_fnames = glob.glob(ndmi_tseries_path.format(data_group))

    for i in range(len(ndvi_tseries_fnames)):
        ndvi_tseries_list.append(pd.read_csv(ndvi_tseries_fnames[i]))
        ndmi_tseries_list.append(pd.read_csv(ndmi_tseries_fnames[i]))
        ndvi_tseries_fnames[i] = (ndvi_tseries_fnames[i].split('\\')[1]).split('S2')[0][:-1]
        ndmi_tseries_fnames[i] = (ndmi_tseries_fnames[i].split('\\')[1]).split('S2')[0][:-1]
        # format the timeseries filenames so that they contain only the ISMN station name or identifier

    for i in range(len(ndvi_tseries_list)):
        for j in range(len(ndvi_tseries_list[i])):
            ndvi_tseries_list[i]['C0/date'][j] = ndvi_tseries_list[i]['C0/date'][j][:10]
            ndmi_tseries_list[i]['C0/date'][j] = ndmi_tseries_list[i]['C0/date'][j][:10]
            # format the date column in the timeseries dataframes

    ndvi_tseries_filt_list = []
    ndvi_tseries_outl_list = []
    ndmi_tseries_filt_list = []
    ndmi_tseries_outl_list = []

    for i in range(len(ndvi_tseries_list)):
        ndvi_tseries_outl_list.append(hampel(ndvi_tseries_list[i]['C0/mean'], window_size=5, n=3))
        ndvi_tseries_filt_list.append(ndvi_tseries_list[i].drop(ndvi_tseries_outl_list[i]))
        ndvi_tseries_outl_list[i] = ndvi_tseries_list[i].iloc[ndvi_tseries_outl_list[i]]
        # create a list where for each ISMN station the NDVI filtered dataframes (with the use of the Hampel filter)
        # will be stored as well as dataframes containing only the resulting outliers
        ndmi_tseries_outl_list.append(hampel(ndmi_tseries_list[i]['C0/mean'], window_size=5, n=3))
        ndmi_tseries_filt_list.append(ndmi_tseries_list[i].drop(ndmi_tseries_outl_list[i]))
        ndmi_tseries_outl_list[i] = ndmi_tseries_list[i].iloc[ndmi_tseries_outl_list[i]]
        # create a list where for each ISMN station the NDMI filtered dataframes (with the use of the Hampel filter)
        # will be stored as well as dataframes containing only the resulting outliers

    def cr_tseries_figs(ndvi_tseries, ndmi_tseries, ndvi_tseries_outl, ndmi_tseries_outl,
                        ndvi_res_dir, ndmi_res_dir, ndvi_res_dir_filt, ndmi_res_dir_filt, outl_filt):
        """

        Args:
            ndvi_tseries:
            ndmi_tseries:
            ndvi_tseries_outl:
            ndmi_tseries_outl:
            ndvi_res_dir:
            ndmi_res_dir:
            ndvi_res_dir_filt:
            ndmi_res_dir_filt:
            outl_filt:

        Returns:

        """

        if not os.path.isdir(ndvi_mkdir_filt) and not os.path.isdir(ndmi_mkdir_filt):
            os.mkdir(ndvi_mkdir_filt)
            os.mkdir(ndmi_mkdir_filt)
        elif not os.path.isdir(ndvi_mkdir) and not os.path.isdir(ndmi_mkdir):
            os.mkdir(ndvi_mkdir)
            os.mkdir(ndmi_mkdir)
        else:
            pass
        # create directories for the storing of the results

        ind_stats = pd.DataFrame(columns=['NDVI Mean', 'NDVI Std', 'NDMI Mean', 'NDMI Std'],
                                 index=ndvi_tseries_fnames)
        for i in range(len(ndvi_tseries)):
            # fig, ax_ndvi = plt.subplots(figsize=(15, 10))
            # fig, ax_ndmi = plt.subplots(figsize=(15, 10))

            # ax_ndvi.set_title('NDVI Index Sentinel-2 Timeseries for {} ISMN Station'.format(ndvi_tseries_fnames[i]))
            # ax_ndmi.set_title('NDMI Index Sentinel-2 Timeseries for {} ISMN Station'.format(ndvi_tseries_fnames[i]))
            # ax_ndvi.set_xlabel('Date')
            # ax_ndvi.set_ylabel('NDVI')
            # ax_ndmi.set_xlabel('Date')
            # ax_ndmi.set_ylabel('NDMI')
            # set the title and the labels for the X and Y data for each Sentinel-2 index

            # ax_ndvi.plot(ndvi_tseries[i]['C0/date'], ndvi_tseries[i]['C0/mean'])
            # ax_ndmi.plot(ndmi_tseries[i]['C0/date'], ndmi_tseries[i]['C0/mean'])
            # plot the timeseries for each Sentinel-2 timeseries

            # if outl_filt:
            #     ax_ndvi.scatter(ndvi_tseries_outl[i]['C0/date'], ndvi_tseries_outl[i]['C0/mean'], color='red',
            #                     label='Outlier Points')
            #     ax_ndmi.scatter(ndmi_tseries_outl[i]['C0/date'], ndmi_tseries_outl[i]['C0/mean'], color='red',
            #                     label='Outlier Points')
            #     ax_ndvi.legend()
            #     ax_ndmi.legend()
            # else:
            #     pass
            # plot the outlier points that are generated by the Hampel filtering for each ISMN station

            # ax_ndvi.xaxis.set_major_locator(plt.MaxNLocator(5))
            # ax_ndmi.xaxis.set_major_locator(plt.MaxNLocator(5))

            ind_stats['NDVI Mean'].iloc[i] = np.mean(ndvi_tseries[i]['C0/mean'])
            ind_stats['NDVI Std'].iloc[i] = np.std(ndvi_tseries[i]['C0/mean'])
            ind_stats['NDMI Mean'].iloc[i] = np.mean(ndmi_tseries[i]['C0/mean'])
            ind_stats['NDMI Std'].iloc[i] = np.std(ndmi_tseries[i]['C0/mean'])
            # append the mean values and the standard deviation of the indices to a dataframe

            # ax_ndvi.errorbar(ndvi_tseries[i]['C0/date'], ndvi_tseries[i]['C0/mean'],
            #                  yerr=ndvi_tseries[i]['C0/stDev'], fmt='o', ecolor='green', markersize=6,
            #                  capsize=5)
            # ax_ndmi.errorbar(ndmi_tseries[i]['C0/date'], ndmi_tseries[i]['C0/mean'],
            #                  yerr=ndmi_tseries[i]['C0/stDev'], fmt='o', ecolor='green', markersize=6,
            #                  capsize=5)
            # plot the error (standard deviation) for each data point in the timeseries

            # ax_ndvi.text(0.8, 0.9, r'$\bar x$' + '=' + '{:.2f}'.format(ind_stats['NDVI Mean'][i]),
            #              transform=ax_ndvi.transAxes, fontsize=15)
            # ax_ndmi.text(0.8, 0.9, r'$\bar x$' + '=' + '{:.2f}'.format(ind_stats['NDMI Mean'][i]),
            #              transform=ax_ndmi.transAxes,
            #              fontsize=15)
            #
            # ax_ndvi.text(0.8, 0.85, r'$\sigma$' + '=' + '{:.2f}'.format(ind_stats['NDVI Std'][i]),
            #              transform=ax_ndvi.transAxes,
            #              fontsize=15)
            # ax_ndmi.text(0.8, 0.85, r'$\sigma$' + '=' + '{:.2f}'.format(ind_stats['NDMI Std'][i]),
            #              transform=ax_ndmi.transAxes,
            #              fontsize=15)
            # print in the layout the mean value (x bar) and the standard deviation (sigma)

            # fig.tight_layout()
            # if outl_filt:
            #     ax_ndvi.figure.savefig(
            #         ndvi_res_dir_filt + '/{}_Filtered_NDVI_Timeseries.png'.format(ndvi_tseries_fnames[i]),
            #         dpi=400)
            #     ax_ndmi.figure.savefig(
            #         ndmi_res_dir_filt + '/{}_Filtered_NDMI_Timeseries.png'.format(ndmi_tseries_fnames[i]),
            #         dpi=400)
            # else:
                # ax_ndvi.figure.savefig(ndvi_res_dir + '/{}_NDVI_Timeseries.png'.format(ndvi_tseries_fnames[i]),
                #                        dpi=400)
                # ax_ndmi.figure.savefig(ndmi_res_dir + '/{}_NDMI_Timeseries.png'.format(ndmi_tseries_fnames[i]),
                #                        dpi=400)
            # save the figures in their respective folders

        return ind_stats

    ind_stats_filt = cr_tseries_figs(ndvi_tseries_filt_list, ndmi_tseries_filt_list, ndvi_tseries_outl_list,
                                     ndmi_tseries_outl_list, None, None, ndvi_mkdir_filt, ndmi_mkdir_filt, True)
    ind_stats_in = cr_tseries_figs(ndvi_tseries_list, ndmi_tseries_list, None, None, ndvi_mkdir,
                                   ndmi_mkdir, None, None, None)

    if sar_ims_or_tseries == 'Backscatter Coefficient Timeseries':
        ndvi_dfs_group_1 = [[], []]
        ndvi_dfs_group_2 = [[], []]
        ndmi_dfs_group_1 = [[], []]
        ndmi_dfs_group_2 = [[], []]

        print(ind_stats_filt)
        ndvi_threshold = float(input('Please give the threshold (float number) for the NDVI Timeseries: '))
        ndmi_threshold = float(input('Please give the threshold (float number) for the NDMI timeseries: '))

        for i in range(len(ind_stats_filt)):
            for j in range(len(in_par_dfs)):
                if ind_stats_filt['NDVI Mean'][i] <= ndvi_threshold:
                    ndvi_dfs_group_1[j].append(in_par_dfs[j][i])
                else:
                    ndvi_dfs_group_2[j].append(in_par_dfs[j][i])
                if ind_stats_filt['NDMI Mean'][i] <= ndmi_threshold:
                    ndmi_dfs_group_1[j].append(in_par_dfs[j][i])
                else:
                    ndmi_dfs_group_2[j].append(in_par_dfs[j][i])

        ndvi_dfs_group_1 = [pd.concat(ndvi_dfs_group_1[0]), pd.concat(ndvi_dfs_group_1[1])]
        ndvi_dfs_group_2 = [pd.concat(ndvi_dfs_group_2[0]), pd.concat(ndvi_dfs_group_2[1])]

        ndmi_dfs_group_1 = [pd.concat(ndmi_dfs_group_1[0]), pd.concat(ndmi_dfs_group_1[1])]
        ndmi_dfs_group_2 = [pd.concat(ndmi_dfs_group_2[0]), pd.concat(ndmi_dfs_group_2[1])]
        # group the initial parameters dataframes according to one threshold for each Sentinel-2 index


        # delete possible duplicate dataframe rows if for two given rows there are separate SAR HV and VV backscatter
        # coefficient entries that have the same number of rows and columns

        return ndvi_dfs_group_1, ndvi_dfs_group_2, ndmi_dfs_group_1, ndmi_dfs_group_2

    else:
        ndvi_dfs_group_1 = []
        ndvi_dfs_group_2 = []
        ndmi_dfs_group_1 = []
        ndmi_dfs_group_2 = []

        print(ind_stats_filt)
        ndvi_threshold = float(input('Please give the threshold (float number) for the NDVI Timeseries: '))
        ndmi_threshold = float(input('Please give the threshold (float number) for the NDMI timeseries: '))

        for i in range(len(ind_stats_filt)):
            if ind_stats_filt['NDVI Mean'][i] <= ndvi_threshold:
                ndvi_dfs_group_1.append(in_par_dfs[i])
            else:
                ndvi_dfs_group_2.append(in_par_dfs[i])
            if ind_stats_filt['NDMI Mean'][i] <= ndmi_threshold:
                ndmi_dfs_group_1.append(in_par_dfs[i])
            else:
                ndmi_dfs_group_2.append(in_par_dfs[i])

        grouped_dfs_list_ndvi = [ndvi_dfs_group_1, ndvi_dfs_group_2]
        grouped_dfs_list_ndmi = [ndmi_dfs_group_1, ndmi_dfs_group_2]
        for df_ndvi, df_ndmi in (grouped_dfs_list_ndvi, grouped_dfs_list_ndmi):
            if not df_ndvi:
                grouped_dfs_list_ndvi.remove(df_ndvi)
            if not df_ndmi:
                grouped_dfs_list_ndmi.remove(df_ndmi)

        for i, j in zip(range(len(grouped_dfs_list_ndvi)), range(len(grouped_dfs_list_ndmi))):
            grouped_dfs_list_ndvi[i] = pd.concat(grouped_dfs_list_ndvi[i])
            grouped_dfs_list_ndmi[i] = pd.concat(grouped_dfs_list_ndmi[i])

        return grouped_dfs_list_ndvi, grouped_dfs_list_ndmi
