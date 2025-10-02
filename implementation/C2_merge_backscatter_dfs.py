import pandas as pd
import numpy as np


def merge_bscatter_dfs(l, hrms_vals_per_mod, dub_mod_dfs, nem_mod_dfs, oh_vers_dfs, iem_mod_dfs):
    """
        Args:
        l: the wavelength of the SAR signal
        hrms_vals_per_mod: a tuple containing all the khrms values ranges for each backscatter model and polarization
        emp_mods_dfs: a list of dataframes that contain the S1 acquisitioned and modeled backscatter coefficients
        according to the Dubois and New Empirical Models for a range of Hrms values inside the validity domain of
        each model
        oh_vers_dfs: a list of dataframes that contain the S1 acquisitioned and modeled backscatter coefficients
        in accordance to the two versions of the Oh model from 2002 and 2004 inside the validity domain of the two
        versions of the model
        iem_mod_dfs: a list of dataframes that contain the S1 acquisitioned and modeled backscatter coefficients
        in accordance to the Integral Equation Model as well as  inside the validity domain of the two
        versions of the model
    Returns:

    """
    k = (2 * np.pi) / l
    s0_SAR_HV_mean = np.mean(nem_mod_dfs[0]['s0_SAR_HV_(dB)'])
    s0_SAR_VV_mean = np.mean(dub_mod_dfs[0]['s0_SAR_VV_(dB)'])
    # mean value of the HV and VV polarized S1 S.A.R. backscatter coefficients and wave number of the SAR signal

    s0_Dubois_mean_bias = [abs(np.mean(dub_mod_dfs[i]['s0_Dubois_VV_(dB)']) - s0_SAR_VV_mean)
                           for i in range(len(dub_mod_dfs))]
    s0_NEM_mean_bias = [abs(np.mean(nem_mod_dfs[i]['s0_N.E.M._HV_(dB)']) - s0_SAR_HV_mean)
                        for i in range(len(nem_mod_dfs))], \
                       [abs(np.mean(nem_mod_dfs[i]['s0_N.E.M._VV_(dB)']) - s0_SAR_VV_mean)
                        for i in range(len(nem_mod_dfs))]
    # for each empirical model and polarization, compute the mean biases of the modeled backscatter coefficients
    # and S1 acquisioned backscatter coefficinets for all S1 images acquisition dates and for each Hrms value
    # in a range values inside the validity domain

    s0_Oh2002_mean_bias = [abs(np.mean(oh_vers_dfs[i]['s0_Oh-2002_HV_(dB)']) - s0_SAR_HV_mean)
                           for i in range(len(oh_vers_dfs))], \
                          [abs(np.mean(oh_vers_dfs[i]['s0_Oh-2002_VV_(dB)']) - s0_SAR_VV_mean)
                           for i in range(len(oh_vers_dfs))]
    s0_Oh2004_mean_bias = [abs(np.mean(oh_vers_dfs[i]['s0_Oh-2004_HV_(dB)']) - s0_SAR_HV_mean)
                           for i in range(len(oh_vers_dfs))], \
                          [abs(np.mean(oh_vers_dfs[i]['s0_Oh-2004_VV_(dB)']) - s0_SAR_VV_mean)
                           for i in range(len(oh_vers_dfs))]
    # for each version of the Oh model and polarization, compute the mean biases of the modeled backscatter
    # coefficients and S1 acquisioned backscatter coefficients for all S1 images acquisition dates and for each
    # Hrms value in a range values inside the validity domain

    s0_IEM_gauss_Fung_mean_bias = [[abs(np.mean(iem_mod_dfs[i]['s0_IEM_VV_(dB)_Gauss_ACF_Fung']) -
                                        s0_SAR_VV_mean) for i in range(len(iem_mod_dfs))],
                                   [abs(np.mean(iem_mod_dfs[i]['s0_IEM_B_VV_(dB)_Gauss_ACF_Fung']) -
                                        s0_SAR_VV_mean) for i in range(len(iem_mod_dfs))]]
    s0_IEM_exp_Fung_mean_bias = [[abs(np.mean(iem_mod_dfs[i]['s0_IEM_VV_(dB)_Exp_ACF_Fung']) - s0_SAR_VV_mean)
                                  for i in range(len(iem_mod_dfs))],
                                 [abs(np.mean(iem_mod_dfs[i]['s0_IEM_B_VV_(dB)_Exp_ACF_Fung']) - s0_SAR_VV_mean)
                                  for i in range(len(iem_mod_dfs))]]
    s0_IEM_gauss_Brogioni_mean_bias = [[abs(np.mean(iem_mod_dfs[i]['s0_IEM_VV_(dB)_Gauss_ACF_Brogioni']) -
                                            s0_SAR_VV_mean) for i in range(len(iem_mod_dfs))],
                                       [abs(np.mean(iem_mod_dfs[i]['s0_IEM_B_VV_(dB)_Gauss_ACF_Brogioni']) -
                                            s0_SAR_VV_mean) for i in range(len(iem_mod_dfs))]]
    s0_IEM_exp_Brogioni_mean_bias = [[abs(np.mean(iem_mod_dfs[i]['s0_IEM_VV_(dB)_Exp_ACF_Brogioni']) -
                                          s0_SAR_VV_mean) for i in range(len(iem_mod_dfs))],
                                     [abs(np.mean(iem_mod_dfs[i]['s0_IEM_B_VV_(dB)_Exp_ACF_Brogioni']) -
                                          s0_SAR_VV_mean) for i in range(len(iem_mod_dfs))]]

    s0_IEM_mean_biases_list = [s0_IEM_gauss_Fung_mean_bias, s0_IEM_exp_Fung_mean_bias, s0_IEM_gauss_Brogioni_mean_bias,
                               s0_IEM_exp_Brogioni_mean_bias]
    s0_IEM_mean_biases_list = [[min(s0_IEM_mean_biases_list[i][j]) for j in range(len(s0_IEM_mean_biases_list[i]))] \
                               for i in range(len(s0_IEM_mean_biases_list))]
    # for each ACF type, compute the mean biases of the modeled backscatter coefficients and S1 acquisioned
    # backscatter coefficinets for all S1 images acquisition dates and for each value in a range values inside
    # the validity domain of the Integral Equation Model as well as the Integral Equation Model calibrated by
    # Baghdadi (the calibration introduces a new parametrization of the correlation length as a function of the Hrms
    # and incidence angle and finally store all the mean biases in a list

    khrms_lists = [[k * hrms_vals_per_mod[i][j] for j in range(len(hrms_vals_per_mod[i]))]
                   for i in range(len(hrms_vals_per_mod))]
    # create a list with the 4 sets of kHrms values for the validity domains of each backscatter
    # model

    final_emp_mods_df_cols_list = [nem_mod_dfs[0]['s0_SAR_HV_(dB)'],
                                   dub_mod_dfs[0]['s0_SAR_VV_(dB)'],
                                   dub_mod_dfs[s0_Dubois_mean_bias.index(min(s0_Dubois_mean_bias))]
                                   ['s0_Dubois_VV_(dB)'],
                                   nem_mod_dfs[s0_NEM_mean_bias[0].index(min(s0_NEM_mean_bias[0]))]
                                   ['s0_N.E.M._HV_(dB)'],
                                   nem_mod_dfs[s0_NEM_mean_bias[1].index(min(s0_NEM_mean_bias[1]))]
                                   ['s0_N.E.M._VV_(dB)']]

    best_khrms_vals_ind_emp_models = [khrms_lists[0][s0_Dubois_mean_bias.index(min(s0_Dubois_mean_bias))],
                                      khrms_lists[1][s0_NEM_mean_bias[0].index(min(s0_NEM_mean_bias[0]))],
                                      khrms_lists[1][s0_NEM_mean_bias[1].index(min(s0_NEM_mean_bias[1]))]]
    final_emp_mod_df_cols_heads = [
        's0_Dubois_VV_(dB)' + '_kHrms=' + '{:.2f}'.format(best_khrms_vals_ind_emp_models[0]),
        's0_N.E.M._HV_(dB)' + '_kHrms=' + '{:.2f}'.format(best_khrms_vals_ind_emp_models[1]),
        's0_N.E.M._VV_(dB)' + '_kHrms=' + '{:.2f}'.format(best_khrms_vals_ind_emp_models[2])]
    final_emp_mods_df = pd.concat(final_emp_mods_df_cols_list, axis=1, join='inner')
    final_emp_mods_df.rename(columns={'s0_Dubois_VV_(dB)': final_emp_mod_df_cols_heads[0],
                                      's0_N.E.M._HV_(dB)': final_emp_mod_df_cols_heads[1],
                                      's0_N.E.M._VV_(dB)': final_emp_mod_df_cols_heads[2]}, inplace=True)

    final_oh_vers_df_cols_list = [oh_vers_dfs[s0_Oh2002_mean_bias[0].
        index(min(s0_Oh2002_mean_bias[0]))]['s0_Oh-2002_HV_(dB)'],
                                  oh_vers_dfs[s0_Oh2002_mean_bias[1].
                                      index(min(s0_Oh2002_mean_bias[1]))]['s0_Oh-2002_VV_(dB)'],
                                  oh_vers_dfs[s0_Oh2004_mean_bias[0].
                                      index(min(s0_Oh2004_mean_bias[0]))]['s0_Oh-2004_HV_(dB)'],
                                  oh_vers_dfs[s0_Oh2004_mean_bias[1].
                                      index(min(s0_Oh2004_mean_bias[1]))]['s0_Oh-2004_VV_(dB)']]

    best_khrms_vals_ind_oh_vers = [khrms_lists[2][s0_Oh2002_mean_bias[0].index(min(s0_Oh2002_mean_bias[0]))],
                                   khrms_lists[2][s0_Oh2002_mean_bias[1].index(min(s0_Oh2002_mean_bias[1]))],
                                   khrms_lists[3][s0_Oh2004_mean_bias[0].index(min(s0_Oh2004_mean_bias[0]))],
                                   khrms_lists[3][s0_Oh2004_mean_bias[1].index(min(s0_Oh2004_mean_bias[1]))]]

    final_oh_vers_df_cols_heads = [
        's0_Oh-2002_HV_(dB)' + '_kHrms=' + '{:.2f}'.format(best_khrms_vals_ind_oh_vers[0]),
        's0_Oh-2002_VV_(dB)' + '_kHrms=' + '{:.2f}'.format(best_khrms_vals_ind_oh_vers[1]),
        's0_Oh-2004_HV_(dB)' + '_kHrms=' + '{:.2f}'.format(best_khrms_vals_ind_oh_vers[2]),
        's0_Oh-2004_VV_(dB)' + '_kHrms=' + '{:.2f}'.format(best_khrms_vals_ind_oh_vers[3])]
    final_oh_mods_df = pd.concat(final_oh_vers_df_cols_list, axis=1, join='inner')
    final_oh_mods_df.rename(columns={'s0_Oh-2002_HV_(dB)': final_oh_vers_df_cols_heads[0],
                                     's0_Oh-2002_VV_(dB)': final_oh_vers_df_cols_heads[1],
                                     's0_Oh-2004_HV_(dB)': final_oh_vers_df_cols_heads[2],
                                     's0_Oh-2004_VV_(dB)': final_oh_vers_df_cols_heads[3]}, inplace=True)

    final_iem_mod_df_cols_list = [iem_mod_dfs[0]['s0_SAR_VV_(dB)'],
                                  iem_mod_dfs[s0_IEM_gauss_Fung_mean_bias[0].index(
                                      min(s0_IEM_gauss_Fung_mean_bias[0]))]['s0_IEM_VV_(dB)_Gauss_ACF_Fung'],
                                  iem_mod_dfs[s0_IEM_exp_Fung_mean_bias[0].index(
                                      min(s0_IEM_exp_Fung_mean_bias[0]))]['s0_IEM_VV_(dB)_Exp_ACF_Fung'],
                                  iem_mod_dfs[s0_IEM_gauss_Brogioni_mean_bias[0].index(
                                      min(s0_IEM_gauss_Brogioni_mean_bias[0]))]['s0_IEM_VV_(dB)_Gauss_ACF_Brogioni'],
                                  iem_mod_dfs[s0_IEM_exp_Brogioni_mean_bias[0].index(
                                      min(s0_IEM_exp_Brogioni_mean_bias[0]))]['s0_IEM_VV_(dB)_Exp_ACF_Brogioni'],
                                  iem_mod_dfs[s0_IEM_gauss_Fung_mean_bias[1].index(
                                      min(s0_IEM_gauss_Fung_mean_bias[1]))]['s0_IEM_B_VV_(dB)_Gauss_ACF_Fung'],
                                  iem_mod_dfs[s0_IEM_exp_Fung_mean_bias[1].index(
                                      min(s0_IEM_exp_Fung_mean_bias[1]))]['s0_IEM_B_VV_(dB)_Exp_ACF_Fung'],
                                  iem_mod_dfs[s0_IEM_gauss_Brogioni_mean_bias[1].index(
                                      min(s0_IEM_gauss_Brogioni_mean_bias[1]))]['s0_IEM_B_VV_(dB)_Gauss_ACF_Brogioni'],
                                  iem_mod_dfs[s0_IEM_exp_Brogioni_mean_bias[1].index(
                                      min(s0_IEM_exp_Brogioni_mean_bias[1]))]['s0_IEM_B_VV_(dB)_Exp_ACF_Brogioni']]

    best_khrms_vals_ind_iem = [khrms_lists[4][s0_IEM_gauss_Fung_mean_bias[0].index(
        min(s0_IEM_gauss_Fung_mean_bias[0]))],
                               khrms_lists[4][s0_IEM_exp_Fung_mean_bias[0].index(
                                   min(s0_IEM_exp_Fung_mean_bias[0]))],
                               khrms_lists[4][s0_IEM_gauss_Brogioni_mean_bias[0].index(
                                   min(s0_IEM_gauss_Brogioni_mean_bias[0]))],
                               khrms_lists[4][s0_IEM_exp_Brogioni_mean_bias[0].index(
                                   min(s0_IEM_exp_Brogioni_mean_bias[0]))],
                               khrms_lists[4][s0_IEM_gauss_Fung_mean_bias[1].index(
                                   min(s0_IEM_gauss_Fung_mean_bias[1]))],
                               khrms_lists[4][s0_IEM_exp_Fung_mean_bias[1].index(
                                   min(s0_IEM_exp_Fung_mean_bias[1]))],
                               khrms_lists[4][s0_IEM_gauss_Brogioni_mean_bias[1].index(
                                   min(s0_IEM_gauss_Brogioni_mean_bias[1]))],
                               khrms_lists[4][s0_IEM_exp_Brogioni_mean_bias[1].index(
                                   min(s0_IEM_exp_Brogioni_mean_bias[1]))]]

    final_iem_mod_df_cols_heads = [
        's0_IEM_VV_(dB)_Gauss_ACF_Fung' + '_kHrms=' + '{:.2f}'.format(best_khrms_vals_ind_iem[0]),
        's0_IEM_VV_(dB)_Exp_ACF_Fung' + '_kHrms=' + '{:.2f}'.format(best_khrms_vals_ind_iem[1]),
        's0_IEM_VV_(dB)_Gauss_ACF_Brogioni' + '_kHrms=' + '{:.2f}'.format(best_khrms_vals_ind_iem[2]),
        's0_IEM_VV_(dB)_Exp_ACF_Brogioni' + '_kHrms=' + '{:.2f}'.format(best_khrms_vals_ind_iem[3]),
        's0_IEM_B_VV_(dB)_Gauss_ACF_Fung' + '_kHrms=' + '{:.2f}'.format(best_khrms_vals_ind_iem[4]),
        's0_IEM_B_VV_(dB)_Exp_ACF_Fung' + '_kHrms=' + '{:.2f}'.format(best_khrms_vals_ind_iem[5]),
        's0_IEM_B_VV_(dB)_Gauss_ACF_Brogioni' + '_kHrms=' + '{:.2f}'.format(best_khrms_vals_ind_iem[6]),
        's0_IEM_B_VV_(dB)_Exp_ACF_Brogioni' + '_kHrms=' + '{:.2f}'.format(best_khrms_vals_ind_iem[7])]

    [s0_IEM_mean_biases_list[i].insert(0, final_iem_mod_df_cols_heads[i]) for i in range(len(s0_IEM_mean_biases_list))]
    [s0_IEM_mean_biases_list[i].insert(2, final_iem_mod_df_cols_heads[i+4]) for i in range(len(s0_IEM_mean_biases_list))]
    s0_IEM_mean_biases_list = s0_IEM_mean_biases_list[0] + s0_IEM_mean_biases_list[1] + s0_IEM_mean_biases_list[2] + \
                              s0_IEM_mean_biases_list[3]
    # insert into the mean biases lists the column headeris for each ACF or revision of the I.E.M. model

    final_iem_mod_df = pd.concat(final_iem_mod_df_cols_list, axis=1, join='inner')
    final_iem_mod_df.rename(columns={'s0_IEM_VV_(dB)_Gauss_ACF_Fung': final_iem_mod_df_cols_heads[0],
                                     's0_IEM_VV_(dB)_Exp_ACF_Fung': final_iem_mod_df_cols_heads[1],
                                     's0_IEM_VV_(dB)_Gauss_ACF_Brogioni': final_iem_mod_df_cols_heads[2],
                                     's0_IEM_VV_(dB)_Exp_ACF_Brogioni': final_iem_mod_df_cols_heads[3],
                                     's0_IEM_B_VV_(dB)_Gauss_ACF_Fung': final_iem_mod_df_cols_heads[4],
                                     's0_IEM_B_VV_(dB)_Exp_ACF_Fung': final_iem_mod_df_cols_heads[5],
                                     's0_IEM_B_VV_(dB)_Gauss_ACF_Brogioni': final_iem_mod_df_cols_heads[6],
                                     's0_IEM_B_VV_(dB)_Exp_ACF_Brogioni': final_iem_mod_df_cols_heads[7]},
                            inplace=True)

    return final_emp_mods_df, final_oh_mods_df, final_iem_mod_df, s0_IEM_mean_biases_list
