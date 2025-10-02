import itertools

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew


def filter_outliers(models_df, in_par_df, mods_str_VV, outl_pdf_plots_path, bef_filt_sc_plots_path,
                    after_filt_sc_plots_path, bef_filt_tseries_path, aft_filt_tseries_path):
    """

    Args:
        models_df: a dataframe with the SAR acquired and modeled sigma0 values for each polarization
        in_par_df: the dataframe with the initial parameters for the computation of the modeled backscatter coefficients
        for each I.S.M.N. station
        mods_str_VV: a list with the names as strings of each physical or empirical backscatter model for which a
        backscatter coefficient in the VV polarization can be computed (needed for the generation of the titles of the
        scatter plots and the time series of the backscatter coefficient values for each date for which respective data
        was acquired)
        outl_pdf_plots_path: the path string to the folder where the standard deviation bell curves for each modeled and
        SAR acquired backscatter coefficient dataset will be stored
        bef_filt_sc_plots_path: the path to the folder where the comparison plots before the outlier filtering (through
        the plotting of the best fit line) between the SAR acquired and the modeled backscatter coefficients will be stored
        after_filt_sc_plots_path: the path to the folder where the comparison plots after the outlier filtering (through
        the plotting of the best fit line) between the SAR acquired and the modeled backscatter coefficients will be stored
        bef_filt_tseries_path: the path to the folder where the time series before the outlier filtering between the SAR
        acquired and the modeled backscatter coefficients will be stored
        aft_filt_tseries_path: the path to the folder where the comparison plots after the outlier filtering (through
        the plotting of the best fit line) between the SAR acquired and the modeled backscatter coefficients will be stored

    Returns: sar_mod_list, a list which contains dataframes with the sets of the modeled and SAR acquired backscatter
    coefficients for each model and polarization for the I.S.M.N. station

    """

    col_all_names = list(models_df.columns)
    col_mod_names = col_all_names[2:]

    df_means = [np.mean(models_df[col][models_df[col].notna()]) for col in col_all_names]
    df_medians = [np.median(models_df[col][models_df[col].notna()]) for col in col_all_names]
    df_skew = [skew(models_df[col][models_df[col].notna()]) for col in col_all_names]
    df_std = [np.std(models_df[col][models_df[col].notna()]) for col in col_all_names]
    df_pdf = [(1 / (df_std[i] * np.sqrt(np.pi * 2))) * np.exp(-0.5 * ((models_df[col][models_df[col].notna()] -
              df_means[i]) / df_std[i]) ** 2) for col, i in zip(col_all_names, range(len(col_all_names)))]
    # compute some statistical indices for normality checking

    sar_mods_str = ['S.A.R. HV', 'S.A.R. VV', 'Dubois VV', 'N.E.M. HV', 'N.E.M. VV', 'Oh 2002 HV', 'Oh 2002 VV',
                    'Oh 2004 HV', 'Oh 2004 VV', 'I.E.M. Fung Gaussian A.C.F.', 'I.E.M. Fung Exponential A.C.F.',
                    'I.E.M. Brogioni Gaussian A.C.F.', 'I.E.M. Brogioni Exponential A.C.F',
                    'I.E.M. Baghdadi Calibrated Fung Gaussian A.C.F.',
                    'I.E.M. Baghdadi Calibrated Fung Exponential A.C.F.',
                    'I.E.M. Baghdadi Calibrated Brogioni Gaussian A.C.F.',
                    'I.E.M. Baghdadi Calibrated Brogioni Exponential A.C.F.']
    station_name = input('Enter the name of the I.S.M.N. Station: ')
    station_suff = input('Please enter a suffix for the I.S.M.N. Station: ')

    khrms_mods_str = ['', ''] + [col[-10:] for col in col_mod_names]

    for i, col, sar_mod_str, khrms_mod in zip(range(len(col_all_names)), col_all_names, sar_mods_str, khrms_mods_str):
        fig, ax = plt.subplots(figsize=(11, 11))
        ax.set_title('Probability Density Function Plot for ' + r'$\sigma_0$' +
                     ' {} Polarization'.format(sar_mod_str) + '\n and {}'.format(khrms_mod) + ' for {} Station'.
                     format(station_name))
        ax.set_xlabel(r'$\sigma_0$' + ' ' + sar_mod_str)
        ax.set_ylabel('PDF')
        ax.scatter(models_df[col][models_df[col].notna()], df_pdf[i])

        ax.plot(models_df[col].sort_values(), (1 / (df_std[i] * np.sqrt(np.pi * 2))) *
                np.exp(-0.5 * ((models_df[col].sort_values() - df_means[i]) / df_std[i]) ** 2))
        ax.text(0.8, 0.8, r'$\mu$' + '={:.2f}'.format(df_means[i]), transform=ax.transAxes, fontsize=12)
        ax.text(0.8, 0.75, 'median={:.2f}'.format(df_medians[i]), transform=ax.transAxes, fontsize=12)
        ax.text(0.8, 0.7, r'$G_1$' + '={:.2f}'.format(df_skew[i]), transform=ax.transAxes, fontsize=12)

        fig.tight_layout()
        fig.savefig(outl_pdf_plots_path + '/pdf_plot_' + col + '_{}'.format(station_suff) + '.png')
        # also plot the pdf's (probability density function) to check visually for normality and then save the plots

    df_q1 = [np.nanquantile(models_df[col], 0.25) for col in col_mod_names]
    df_q3 = [np.nanquantile(models_df[col], 0.75) for col in col_mod_names]
    df_iqr = [df_q3[i] - df_q1[i] for i in range(len(col_mod_names))]
    df_min_lim = [df_q1[i] - 1.5 * df_iqr[i] if abs(df_medians[2:][i] - df_means[2:][i]) > 0.1 and df_skew[2:][i] > 0.1
                  else df_means[2:][i] - 3 * df_std[2:][i] for i in range(len(col_mod_names))]
    df_max_lim = [df_q3[i] + 1.5 * df_iqr[i] if abs(df_medians[2:][i] - df_means[2:][i]) > 0.1 and df_skew[2:][i] > 0.1
                  else df_means[2:][i] + 3 * df_std[2:][i] for i in range(len(col_mod_names))]
    # compute the 25% and 75% quartiles of the distribution of values and then the Interquartile Range
    # (I.Q.R.) in order to find the minimum and maxiumum limits which serve as thresholds for outlier
    # detection if certain criteria are met (skewness or difference between median and mean less than 0.1)

    init_or_filt_dfs = input('Please enter y/n if the dataframes will be used for the outlier values removal: ')
    assert init_or_filt_dfs in ['y', 'n']

    model_series_cols_in = [models_df[col].to_frame() for col in col_mod_names]
    sar_series_cols_in = [models_df[col].to_frame() for col in col_all_names[:2]]

    if 'Precip_(mm)' in in_par_df.columns.to_list():
        precip_cols = in_par_df['Precip_(mm)']
        precip_cols = precip_cols.loc[lambda x: x != 0]
    if 'Below_Freezing_Temp' in in_par_df.columns.to_list():
        below_freeze_temp_cols = in_par_df['Below_Freezing_Temp']
        below_freeze_temp_cols = below_freeze_temp_cols.loc[lambda x: x == True]

    if init_or_filt_dfs == 'n':
        sar_series_cols_in = [[sar_series_cols_in[0], sar_series_cols_in[1]] for i in range(len(model_series_cols_in))]
        if 'Precip_(mm)' in in_par_df.columns.to_list():
            sar_series_precip_in = [[sar_series_cols_in[i][0][sar_series_cols_in[i][0].index.isin(precip_cols.index)],
                                     sar_series_cols_in[i][1][sar_series_cols_in[i][1].index.isin(precip_cols.index)]]
                                    for i in range(len(model_series_cols_in))]
            mod_series_precip_in = [model_series_cols_in[i][model_series_cols_in[i].index.isin(precip_cols.index)]
                                    for i in range(len(model_series_cols_in))]
        if 'Below_Freezing_Temp' in in_par_df.columns.to_list():
            sar_series_below_freeze_in = [
                [sar_series_cols_in[i][0][sar_series_cols_in[i][0].index.isin(below_freeze_temp_cols.
                                                                              index)],
                 sar_series_cols_in[i][1][sar_series_cols_in[i][1].index.isin(
                     below_freeze_temp_cols.index)]] for i in range(len(model_series_cols_in))]
            mod_series_below_freeze_in = [
                model_series_cols_in[i][model_series_cols_in[i].index.isin(below_freeze_temp_cols.
                                                                           index)] for i in
                range(len(model_series_cols_in))]
    elif init_or_filt_dfs == 'y':
        model_series_cols_out_filt = [mod_series[(mod_series < max_lim) & (mod_series > min_lim)] for
                                      mod_series, max_lim, min_lim in
                                      zip(model_series_cols_in, df_max_lim, df_min_lim)]
        sar_series_cols_out_filt = [[sar_series_cols_in[j][sar_series_cols_in[j].index.isin(
            model_series_cols_out_filt[i].index)] for j in range(len(sar_series_cols_in))]
                                    for i in range(len(model_series_cols_in))]
        if 'Precip_(mm)' in in_par_df.columns.to_list():
            mod_series_precip_filt = [model_series_cols_out_filt[i][model_series_cols_out_filt[i].index.
                isin(precip_cols.index)] for i in range(len(model_series_cols_out_filt))]
            mod_series_precip_filt = [mod_series_precip_filt[i].dropna() for i in range(len(mod_series_precip_filt))]
            sar_series_precip_filt = [[sar_series_cols_out_filt[i][j][sar_series_cols_out_filt[i][j].index.
                isin(precip_cols.index)] for j in range(len(sar_series_cols_out_filt[i]))]
                                      for i in range(len(sar_series_cols_out_filt))]
        if 'Below_Freezing_Temp' in in_par_df.columns.to_list():
            mod_series_below_freeze_filt = [model_series_cols_out_filt[i][model_series_cols_out_filt[i].index.
                isin(below_freeze_temp_cols.index)] for i in range(len(model_series_cols_out_filt))]
            mod_series_below_freeze_filt = [mod_series_below_freeze_filt[i].dropna()
                                            for i in range(len(mod_series_below_freeze_filt))]
            sar_series_below_freeze_filt = [[sar_series_cols_out_filt[i][j][sar_series_cols_out_filt[i][j].index.
                isin(below_freeze_temp_cols.index)] for j in range(len(sar_series_cols_out_filt[i]))]
                                            for i in range(len(sar_series_cols_out_filt))]

    precip = input('Is there a column in the initial parameters dataframe containing Precipitation data?: ')
    below_freezing_temp = input('Is there a column in the initial parameters dataframe containing booleans indicating a'
                                ' temperature below freezing?: ')

    def cr_filt_rem_figs(sar_series_list, sar_precip, sar_below_freeze, model_series_list, mod_precip, mod_below_freeze,
                         bef_aft_titl_str, bef_aft_savefig_str, filt_sc_plots_path, filt_tseries_path, models_str_VV):
        """

        Args:
            sar_series_list:
            sar_precip:
            model_series_list:
            mod_precip:
            bef_aft_titl_str:
            bef_aft_savefig_str:
            filt_sc_plots_path:
            filt_tseries_path:

        Returns:

        """
        assert bef_aft_titl_str in ['Before', 'After']
        assert bef_aft_savefig_str in ['bef', 'aft']

        sar_mod_list = []
        sar_mod_precip_list = []
        sar_mod_below_freeze_list = []

        for i in range(len(model_series_list)):
            if 'HV' in list(model_series_list[i].columns)[0]:
                sar_mod_list.append(sar_series_list[i][0].merge(model_series_list[i], left_index=True,
                                                                right_index=True))
                sar_mod_list[i].dropna(inplace=True)
                if sar_precip is not None and mod_precip is not None:
                    sar_mod_precip_list.append(sar_precip[i][0].merge(mod_precip[i], left_index=True, right_index=True))
                    sar_mod_precip_list[i].dropna(inplace=True)
                if sar_below_freeze is not None and mod_below_freeze is not None:
                    sar_mod_below_freeze_list.append(sar_below_freeze[i][0].merge(mod_below_freeze[i], left_index=True,
                                                                                  right_index=True))

            elif 'VV' in list(model_series_list[i].columns)[0]:
                sar_mod_list.append(sar_series_list[i][1].merge(model_series_list[i], left_index=True,
                                                                right_index=True))
                sar_mod_list[i].dropna(inplace=True)
                if sar_precip is not None and mod_precip is not None:
                    sar_mod_precip_list.append(sar_precip[i][1].merge(mod_precip[i], left_index=True, right_index=True))
                    sar_mod_precip_list[i].dropna(inplace=True)
                if sar_below_freeze is not None and mod_below_freeze is not None:
                    sar_mod_below_freeze_list.append(sar_below_freeze[i][1].merge(mod_below_freeze[i], left_index=True,
                                                                                  right_index=True))
                    sar_mod_below_freeze_list[i].dropna(inplace=True)
            # create a list of dataframes that does not contain outliers (as outliers are thought these data points that
            # lay more than the maximum and minimum limit distances from the I.Q.R. or the mean if the data are normally
            # distributed)

        models_str_HV = models_str_VV[1:4]
        khrms_mods_str_HV = [sar_mod_list[i].columns[1][-10:] for i in range(1, 6, 2)]
        khrms_mods_str_VV = [sar_mod_list[i].columns[1][-10:] for i in range(len(sar_mod_list)) if 'VV' in
                             list(sar_mod_list[i].columns)[0]]

        # initialize the needed lists for the creation of the scatter plots of the modeled and SAR acquired backscatter
        # coefficients before and after the removal of outlier values

        def comp_rsq(x_ax_data, y_ax_data, a_sl, b_intr):
            y_pred = a_sl * x_ax_data + b_intr
            res_sum_sq = np.sum((y_ax_data - y_pred) ** 2)
            tot_sum_sq = np.sum((y_ax_data - np.mean(y_ax_data)) ** 2)
            r_sq = 1 - res_sum_sq / tot_sum_sq
            return r_sq
            # function for the computation of the R squared statistical coefficient of the modeled and SAR acquired
            # backscatter coefficients dataset

        def comp_rmse(x_ax_data, y_ax_data, a_sl, b_intr):
            y_pred = a_sl * x_ax_data + b_intr
            rmse = np.sqrt(np.sum((y_pred - y_ax_data) ** 2 / len(y_ax_data)))
            return rmse
            # function for the computation of the Root Mean Square error of the modeled and SAR acquired backscatter
            # coefficients dataset

        sar_mod_list_hv_index = []
        sar_mod_list_vv_index = []
        mod_cols_list_hv = []
        mod_cols_list_vv = []

        for i in range(len(sar_mod_list)):
            if 'HV' in sar_mod_list[i].columns[0]:
                sar_mod_list_hv_index.append(i)
                mod_cols_list_hv.append(sar_mod_list[i].columns[1])
            elif 'VV' in sar_mod_list[i].columns[0]:
                sar_mod_list_vv_index.append(i)
                mod_cols_list_vv.append(sar_mod_list[i].columns[1])

        for i_HV, i_VV, khrms_mod_HV, khrms_mod_VV, mod_HV, mod_VV, col_HV, col_VV in itertools.zip_longest(
                sar_mod_list_hv_index,
                sar_mod_list_vv_index,
                khrms_mods_str_HV,
                khrms_mods_str_VV,
                models_str_HV,
                models_str_VV,
                mod_cols_list_hv,
                mod_cols_list_vv):

            def HV_pol_scatter_plots(i_HV, khrms_mod_HV, mod_HV, col_HV):
                fig, ax = plt.subplots(figsize=(11, 11))

                ax.set_title(
                    'Scatter Plot ' + bef_aft_titl_str + ' the Removal of Outliers for {} Model and S.A.R. Acquired\n'.
                    format(mod_HV) + r'$\sigma_0$' + '- HV Polarization' + ' and {}'.format(khrms_mod_HV))
                ax.set_xlabel(r'$\sigma_0$' + '[dB] HV SAR')
                ax.set_ylabel(r'$\sigma_0$' + '[dB] HV {} Model'.format(mod_HV))

                ax.scatter(sar_mod_list[i_HV][sar_mod_list[i_HV].columns[0]],
                           sar_mod_list[i_HV][sar_mod_list[i_HV].columns[1]], color='blue')
                if sar_precip is not None and mod_precip is not None:
                    ax.scatter(sar_mod_precip_list[i_HV][sar_mod_precip_list[i_HV].columns[0]],
                               sar_mod_precip_list[i_HV][sar_mod_precip_list[i_HV].columns[1]], color='red',
                               label='Precipitation Event')
                    ax.legend(loc='upper left')
                if sar_below_freeze is not None and mod_below_freeze is not None:
                    ax.scatter(sar_mod_below_freeze_list[i_HV][sar_mod_below_freeze_list[i_HV].columns[0]],
                               sar_mod_below_freeze_list[i_HV][sar_mod_below_freeze_list[i_HV].columns[1]],
                               color='green', label='Below Freezing Temperature')
                    ax.legend(loc='upper left')

                a, b = np.polyfit(sar_mod_list[i_HV][sar_mod_list[i_HV].columns[0]],
                                  sar_mod_list[i_HV][sar_mod_list[i_HV].columns[1]], deg=1)
                r_sq_hv = comp_rsq(sar_mod_list[i_HV][sar_mod_list[i_HV].columns[0]],
                                   sar_mod_list[i_HV][sar_mod_list[i_HV].columns[1]], a_sl=a, b_intr=b)
                rmse_hv = comp_rmse(sar_mod_list[i_HV][sar_mod_list[i_HV].columns[0]],
                                    sar_mod_list[i_HV][sar_mod_list[i_HV].columns[1]], a_sl=a, b_intr=b)

                ax.plot(sar_mod_list[i_HV][sar_mod_list[i_HV].columns[0]], a * sar_mod_list[i_HV][sar_mod_list[i_HV].
                        columns[0]] + b, color='red')

                ax.text(0.8, 0.8, 'y=' + '{:.2f}'.format(a) + 'x+' + '{:.2f}'.format(b), transform=ax.transAxes,
                        fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="cyan", ec="b", lw=2))

                ax.text(0.8, 0.7, r'$R^2$' + '=' + '{:.2f}'.format(r_sq_hv), transform=ax.transAxes, fontsize=12,
                        bbox=dict(boxstyle="round,pad=0.3", fc="cyan", ec="b", lw=2))
                ax.text(0.8, 0.75, 'RMSE' + '=' + '{:.2f}'.format(rmse_hv), transform=ax.transAxes, fontsize=12,
                        bbox=dict(boxstyle="round,pad=0.3", fc="cyan", ec="b", lw=2))

                ax.legend(loc='upper left')
                fig.subplots_adjust(hspace=1)
                fig.tight_layout()
                ax.figure.savefig(
                    filt_sc_plots_path + '/scatter_plot_' + bef_aft_savefig_str + '_outl_filt_' + col_HV + '.png')

            def HV_pol_time_series(i_HV, khrms_mod_HV, mod_HV, col_HV):
                fig, ax = plt.subplots(figsize=(20, 11))

                ax.set_title(
                    'Time Series ' + bef_aft_titl_str + ' the Removal of Outliers for {} Model and S.A.R. Acquired\n'.
                    format(mod_HV) + r'$\sigma_0$' + '- HV Polarization' + ' and {}'.format(khrms_mod_HV))
                ax.set_xlabel('DOY')
                ax.set_ylabel(r'$\sigma_0$' + '[dB] HV {} Model'.format(mod_HV))

                ax.xaxis.set_major_locator(plt.MaxNLocator(5))

                ax.plot(sar_mod_list[i_HV].index, sar_mod_list[i_HV][sar_mod_list[i_HV].columns[0]], color='blue',
                        label='SAR Acquired '+r'$\sigma_0$')
                ax.plot(sar_mod_list[i_HV].index, sar_mod_list[i_HV][sar_mod_list[i_HV].columns[1]], color='red',
                        label='Modeled '+r'$\sigma_0$')

                if sar_precip is not None and mod_precip is not None:
                    ax.scatter(sar_mod_precip_list[i_HV].index, sar_mod_precip_list[i_HV][sar_mod_precip_list[i_HV].
                               columns[1]], color='red', label='Precipitation Event', marker='s')
                    ax.legend(loc='upper left')
                if sar_below_freeze is not None and mod_below_freeze is not None:
                    ax.scatter(sar_mod_below_freeze_list[i_HV].index, sar_mod_below_freeze_list[i_HV]
                               [sar_mod_below_freeze_list[i_HV].columns[1]], color='green',
                               label='Below Freezing Temperature',  marker='s')
                    ax.legend(loc='upper left')

                ax.legend(loc='upper left')
                fig.subplots_adjust(hspace=1)
                fig.tight_layout()
                ax.figure.savefig(
                    filt_tseries_path + '/time_series_' + bef_aft_savefig_str + '_outl_filt_' + col_HV + '.png')

            def VV_pol_scatter_plots(i_VV, khrms_mod_VV, mod_VV, col_VV):
                fig, ax = plt.subplots(figsize=(11, 11))

                ax.set_title(
                    'Scatter Plot ' + bef_aft_titl_str + ' the Removal of Outliers for {} Model and S.A.R. Acquired\n'.
                    format(mod_VV) + r'$\sigma_0$' + ' - VV Polarization' + ' and {}'.format(khrms_mod_VV))
                ax.set_xlabel(r'$\sigma_0$' + '[dB] VV SAR')
                ax.set_ylabel(r'$\sigma_0$' + '[dB] VV {} Model'.format(mod_VV))

                ax.scatter(sar_mod_list[i_VV][sar_mod_list[i_VV].columns[0]],
                           sar_mod_list[i_VV][sar_mod_list[i_VV].columns[1]], color='blue')
                if sar_precip is not None and mod_precip is not None:
                    ax.scatter(sar_mod_precip_list[i_VV][sar_mod_precip_list[i_VV].columns[0]],
                               sar_mod_precip_list[i_VV][sar_mod_precip_list[i_VV].columns[1]], color='red',
                               label='Precipitation Event')
                    ax.legend(loc='upper left')
                if sar_below_freeze is not None and mod_below_freeze is not None:
                    ax.scatter(sar_mod_below_freeze_list[i_VV][sar_mod_below_freeze_list[i_VV].columns[0]],
                               sar_mod_below_freeze_list[i_VV][sar_mod_below_freeze_list[i_VV].columns[1]],
                               color='green', label='Below Freezing Temperature')
                    ax.legend(loc='upper left')

                a, b = np.polyfit(sar_mod_list[i_VV][sar_mod_list[i_VV].columns[0]],
                                  sar_mod_list[i_VV][sar_mod_list[i_VV].columns[1]], deg=1)
                r_sq_vv = comp_rsq(sar_mod_list[i_VV][sar_mod_list[i_VV].columns[0]],
                                   sar_mod_list[i_VV][sar_mod_list[i_VV].columns[1]], a_sl=a, b_intr=b)
                rmse_vv = comp_rmse(sar_mod_list[i_VV][sar_mod_list[i_VV].columns[0]],
                                    sar_mod_list[i_VV][sar_mod_list[i_VV].columns[1]], a_sl=a, b_intr=b)

                ax.plot(sar_mod_list[i_VV][sar_mod_list[i_VV].columns[0]], a * sar_mod_list[i_VV][sar_mod_list[i_VV].
                        columns[0]] + b, color='red')

                ax.text(0.8, 0.8, 'y=' + '{:.2f}'.format(a) + 'x+' + '{:.2f}'.format(b), transform=ax.transAxes,
                        fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="cyan", ec="b", lw=2))
                ax.text(0.8, 0.7, r'$R^2$' + '=' + '{:.2f}'.format(r_sq_vv), transform=ax.transAxes, fontsize=12,
                        bbox=dict(boxstyle="round,pad=0.3", fc="cyan", ec="b", lw=2))
                ax.text(0.8, 0.75, 'RMSE' + '=' + '{:.2f}'.format(rmse_vv), transform=ax.transAxes, fontsize=12,
                        bbox=dict(boxstyle="round,pad=0.3", fc="cyan", ec="b", lw=2))

                ax.legend(loc='upper left')
                fig.subplots_adjust(hspace=1)
                fig.tight_layout()
                ax.figure.savefig(filt_sc_plots_path + '/scatter_plot_' + bef_aft_savefig_str + '_outl_filt_' + col_VV +
                                  '.png')

            def VV_pol_time_series(i_VV, khrms_mod_VV, mod_VV, col_VV):
                fig, ax = plt.subplots(figsize=(20, 11))

                ax.set_title(
                    'Time Series ' + bef_aft_titl_str + ' the Removal of Outliers for {} Model and S.A.R. Acquired\n'.
                    format(mod_VV) + r'$\sigma_0$' + '- VV Polarization' + ' and {}'.format(khrms_mod_VV))
                ax.set_xlabel('DOY')
                ax.set_ylabel(r'$\sigma_0$' + '[dB] VV {} Model'.format(mod_VV))

                ax.xaxis.set_major_locator(plt.MaxNLocator(5))

                ax.plot(sar_mod_list[i_VV].index, sar_mod_list[i_VV][sar_mod_list[i_VV].columns[0]], color='blue',
                        label='SAR Acquired ' + r'$\sigma_0$')
                ax.plot(sar_mod_list[i_VV].index, sar_mod_list[i_VV][sar_mod_list[i_VV].columns[1]], color='red',
                        label='Modeled ' + r'$\sigma_0$')

                if sar_precip is not None and mod_precip is not None:
                    ax.scatter(sar_mod_precip_list[i_VV].index, sar_mod_precip_list[i_VV][sar_mod_precip_list[i_VV].
                               columns[1]], color='red', label=r'Precipitation Event (Modeled $\sigma_0$)',
                               marker='s')
                    ax.scatter(sar_mod_precip_list[i_VV].index, sar_mod_precip_list[i_VV][sar_mod_precip_list[i_VV].
                               columns[0]], color='blue', label=r'Precipitation Event (SAR Acquired $\sigma_0$)',
                               marker='s')
                    ax.legend(loc='upper left')
                if sar_below_freeze is not None and mod_below_freeze is not None:
                    ax.scatter(sar_mod_below_freeze_list[i_VV].index, sar_mod_below_freeze_list[i_VV]
                              [sar_mod_below_freeze_list[i_VV].columns[1]], color='green',
                               label=r'Below Freezing Temperature (Modeled $\sigma_0$)', marker='s')
                    ax.scatter(sar_mod_below_freeze_list[i_VV].index, sar_mod_below_freeze_list[i_VV]
                              [sar_mod_below_freeze_list[i_VV].columns[0]], color='green',
                               label=r'Below Freezing Temperature (SAR Acquired $\sigma_0$)', marker='s')
                    ax.legend(loc='upper left')

                ax.legend(loc='upper left')
                fig.subplots_adjust(hspace=1)
                fig.tight_layout()
                ax.figure.savefig(
                    filt_tseries_path + '/time_series_' + bef_aft_savefig_str + '_outl_filt_' + col_VV + '.png')

            if i_HV is not None and i_VV is not None:
                HV_pol_scatter_plots(i_HV=i_HV, khrms_mod_HV=khrms_mod_HV, mod_HV=mod_HV, col_HV=col_HV)
                VV_pol_scatter_plots(i_VV=i_VV, khrms_mod_VV=khrms_mod_VV, mod_VV=mod_VV, col_VV=col_VV)

                HV_pol_time_series(i_HV=i_HV, khrms_mod_HV=khrms_mod_HV, mod_HV=mod_HV, col_HV=col_HV)
                VV_pol_time_series(i_VV=i_VV, khrms_mod_VV=khrms_mod_VV, mod_VV=mod_VV, col_VV=col_VV)
            elif i_VV is not None:
                VV_pol_scatter_plots(i_VV=i_VV, khrms_mod_VV=khrms_mod_VV, mod_VV=mod_VV, col_VV=col_VV)
                VV_pol_time_series(i_VV=i_VV, khrms_mod_VV=khrms_mod_VV, mod_VV=mod_VV, col_VV=col_VV)
            elif i_HV is not None:
                HV_pol_scatter_plots(i_HV=i_HV, khrms_mod_HV=khrms_mod_HV, mod_HV=mod_HV, col_HV=col_HV)
                HV_pol_time_series(i_HV=i_HV, khrms_mod_HV=khrms_mod_HV, mod_HV=mod_HV, col_HV=col_HV)
            else:
                pass

        return sar_mod_list,

    if init_or_filt_dfs == 'n':
        if below_freezing_temp == 'y' and precip == 'y':
            cr_filt_rem_figs(sar_series_list=sar_series_cols_in, sar_precip=sar_series_precip_in,
                             sar_below_freeze=sar_series_below_freeze_in, model_series_list=model_series_cols_in,
                             mod_precip=mod_series_precip_in, mod_below_freeze=mod_series_below_freeze_in,
                             bef_aft_titl_str='Before', bef_aft_savefig_str='bef',
                             filt_sc_plots_path=bef_filt_sc_plots_path, filt_tseries_path=bef_filt_tseries_path,
                             models_str_VV=mods_str_VV)
        elif below_freezing_temp == 'n' and precip == 'y':
            cr_filt_rem_figs(sar_series_list=sar_series_cols_in, sar_precip=sar_series_precip_in,
                             sar_below_freeze=None, model_series_list=model_series_cols_in,
                             mod_precip=mod_series_precip_in, mod_below_freeze=None,
                             bef_aft_titl_str='Before', bef_aft_savefig_str='bef',
                             filt_sc_plots_path=bef_filt_sc_plots_path, filt_tseries_path=bef_filt_tseries_path,
                             models_str_VV=mods_str_VV)
        elif below_freezing_temp == 'y' and precip == 'n':
            cr_filt_rem_figs(sar_series_list=sar_series_cols_in, sar_precip=None,
                             sar_below_freeze=sar_series_below_freeze_in, model_series_list=model_series_cols_in,
                             mod_precip=None, mod_below_freeze=mod_series_below_freeze_in,
                             bef_aft_titl_str='Before', bef_aft_savefig_str='bef',
                             filt_sc_plots_path=bef_filt_sc_plots_path, filt_tseries_path=bef_filt_tseries_path,
                             models_str_VV=mods_str_VV)
        else:
            cr_filt_rem_figs(sar_series_list=sar_series_cols_in, sar_precip=None,
                             sar_below_freeze=None, model_series_list=model_series_cols_in,
                             mod_precip=None, mod_below_freeze=None,
                             bef_aft_titl_str='Before', bef_aft_savefig_str='bef',
                             filt_sc_plots_path=bef_filt_sc_plots_path, filt_tseries_path=bef_filt_tseries_path,
                             models_str_VV=mods_str_VV)
    elif init_or_filt_dfs == 'y':
        if below_freezing_temp == 'y' and precip == 'y':
            fin_sar_mod_list = cr_filt_rem_figs(sar_series_list=sar_series_cols_out_filt,
                                                sar_precip=sar_series_precip_filt,
                                                sar_below_freeze=sar_series_below_freeze_filt,
                                                model_series_list=model_series_cols_out_filt,
                                                mod_precip=mod_series_precip_filt,
                                                mod_below_freeze=mod_series_below_freeze_filt,
                                                bef_aft_titl_str='Before', bef_aft_savefig_str='bef',
                                                filt_sc_plots_path=after_filt_sc_plots_path,
                                                filt_tseries_path=aft_filt_tseries_path,
                                                models_str_VV=mods_str_VV)
        elif below_freezing_temp == 'n' and precip == 'y':
            fin_sar_mod_list = cr_filt_rem_figs(sar_series_list=sar_series_cols_out_filt,
                                                sar_precip=sar_series_precip_filt,
                                                sar_below_freeze=None, model_series_list=model_series_cols_out_filt,
                                                mod_precip=mod_series_precip_filt, mod_below_freeze=None,
                                                bef_aft_titl_str='After', bef_aft_savefig_str='aft',
                                                filt_sc_plots_path=after_filt_sc_plots_path,
                                                filt_tseries_path=aft_filt_tseries_path,
                                                models_str_VV=mods_str_VV)
        elif below_freezing_temp == 'y' and precip == 'n':
            fin_sar_mod_list = cr_filt_rem_figs(sar_series_list=sar_series_cols_out_filt, sar_precip=None,
                                                sar_below_freeze=sar_series_below_freeze_filt,
                                                model_series_list=model_series_cols_out_filt,
                                                mod_precip=None, mod_below_freeze=mod_series_below_freeze_filt,
                                                bef_aft_titl_str='After', bef_aft_savefig_str='aft',
                                                filt_sc_plots_path=after_filt_sc_plots_path,
                                                filt_tseries_path=aft_filt_tseries_path,
                                                models_str_VV=mods_str_VV)
        else:
            fin_sar_mod_list = cr_filt_rem_figs(sar_series_list=sar_series_cols_out_filt, sar_precip=None,
                                                sar_below_freeze=None, model_series_list=model_series_cols_out_filt,
                                                mod_precip=None, mod_below_freeze=None,
                                                bef_aft_titl_str='After', bef_aft_savefig_str='aft',
                                                filt_sc_plots_path=after_filt_sc_plots_path,
                                                filt_tseries_path=aft_filt_tseries_path,
                                                models_str_VV=mods_str_VV)

        return fin_sar_mod_list
