import pandas as pd
import numpy as np
from scipy import optimize
from scipy.optimize import NonlinearConstraint
import matplotlib.pyplot as plt
from varname import nameof
from C1_backscatter_models_dataframes import comp_mod_dfs


def inv(filt_mod_dfs, in_par_df, st_clay_pct, st_sand_pct, sar_freq):
    """

    Args:
        filt_mod_dfs: the list which contains the dataframes with the filtered SAR acquired and modeled backscatter
        coefficients for each I.S.M.N. station
        in_par_df: the dataframe with the initial values that are needed for the computation of the modeled backscatter
        coefficient values for each I.S.M.N. statiοn
        st_clay_pct: the clay percentage of the soil in the I.S.M.N. station
        st_sand_pct: the sand percentage of the soil in the I.S.M.N. station
        sar_freq: the frequency of the SAR radar system used in GHz

    Returns: mod_est_sm_list, a list containing dataframes for each model where the modeled and insitu measured soil
    moisture values are contained
    """

    filt_mod_dfs = filt_mod_dfs[1:]

    l_sar = 5.5465763
    k_sar = (2 * np.pi) / l_sar
    theta = in_par_df['Inc_Ang_(rad)'][0]
    dates_list = [filt_mod_dfs[i].index.to_list() for i in range(len(filt_mod_dfs))]
    khrms_vals = [np.float64(filt_mod_dfs[i].columns[1][-4:]) for i in range(len(filt_mod_dfs))]
    hrms_vals = [khrms_vals[i] / k_sar for i in range(len(filt_mod_dfs))]
    l_corr = [(1 / np.e) * khrms_vals[i] ** 2 for i in range(len(khrms_vals))]
    # import and compute all the variables needed for the volumetric soil moisture retrieval using the modeled and the
    # SAR acquired backscatter coefficients

    sm_per_mod = [in_par_df['SM_(m^3/m^3)'][in_par_df['SM_(m^3/m^3)'].index.isin(dates_list[i])]
                  for i in range(len(filt_mod_dfs))]
    sm_statistics = [sm_per_mod[i].describe() for i in range(len(sm_per_mod))]
    sm_per_mod = [sm_series.to_numpy() for sm_series in sm_per_mod]
    sm = [np.random.uniform(low=0., high=sm_statistics[i].loc['max'], size=(len(sm_per_mod[i])))
          for i in range(len(sm_statistics))]
    sm_bounds = (0, 1)
    bounds_seqs = [[sm_bounds for j in range(len(filt_mod_dfs[i]))] for i in range(len(filt_mod_dfs))]
    cons_per_mod = ({'type': 'ineq', 'fun': lambda x: 0.35 - x}, {'type': 'ineq', 'fun': lambda x: 0.35 - x},
                    {'type': 'ineq', 'fun': lambda x: 0.35 - x}, {'type': 'ineq', 'fun': lambda x: x - 0.03},
                    {'type': 'ineq', 'fun': lambda x: x - 0.03}, {'type': 'ineq', 'fun': lambda x: x - 0.04},
                    {'type': 'ineq', 'fun': lambda x: x - 0.04}, {'type': 'ineq', 'fun': lambda x: 0.35 - x},
                    {'type': 'ineq', 'fun': lambda x: 0.35 - x}, {'type': 'ineq', 'fun': lambda x: 0.35 - x},
                    {'type': 'ineq', 'fun': lambda x: 0.35 - x})
    # compute the initial values for the volumetric soil moisture needed for the inversion procedure as well as the
    # bounds and constraints for each backscatter model

    # def dubois_obj_func(x):
    #     invert_dub_dfs = comp_mod_dfs(SAR_s0_VV=None,
    #                                   SAR_s0_VH=None,
    #                                   l=5.5465763,
    #                                   theta=theta,
    #                                   sm=[x],
    #                                   dates=dates_list[0])
    #     # create an instance of the comp_mod_dfs() class from the "C1_backscatter_models_dataframes.py" script in order
    #     # to invert the Dubois Model
    #
    #     s0_dub_vv = invert_dub_dfs.dubois_mod(hrms_vals=hrms_vals[0], st_sand_pct=st_sand_pct, st_clay_pct=st_clay_pct,
    #                                           sar_freq=sar_freq, invert_mod=True)
    #     s0_sar_vv = filt_mod_dfs[0]['s0_SAR_VV_(dB)'].to_numpy()
    #     cost_dub = np.sum((s0_sar_vv - s0_dub_vv) ** 2)
    #     return cost_dub

    # define the objective function for the Dubois model - VV polarization

    def nem_hv_obj_func(x):
        s0_nem_hv = 10 * np.log10((10 ** -2.325) * (np.cos(theta)) ** (-0.01) * 10 ** (0.011 * (1 / np.tan(theta)) *
                                                                                       100 * x) * khrms_vals[0] ** (
                                          0.44 * np.sin(theta)))
        s0_sar_hv = filt_mod_dfs[0]['s0_SAR_HV_(dB)'].to_numpy()
        cost_nem_hv = np.sum((s0_nem_hv - s0_sar_hv) ** 2)
        return cost_nem_hv

    # define the objective function for the New Empirical model - HV polarization

    def nem_vv_obj_func(x):
        s0_nem_vv = 10 * np.log10((10 ** -1.138) * (np.cos(theta)) ** 1.528 * 10 ** (0.008 * (1 / np.tan(theta)) *
                                                                                     100 * x) * khrms_vals[1] ** (
                                          0.71 * np.sin(theta)))
        s0_sar_vv = filt_mod_dfs[1]['s0_SAR_VV_(dB)'].to_numpy()
        cost_nem_vv = np.sum((s0_nem_vv - s0_sar_vv) ** 2)
        return cost_nem_vv

    # define the objective function for the New Empirical model - VV polarization

    def oh_hv_2002_obj_func(x):
        s0_oh_hv = 10 * np.log10(0.11 * (x ** 0.7) * (np.cos(theta)) ** 2.2 * (1 - np.exp((-0.32) * khrms_vals[2]
                                                                                          ** 1.8)))
        s0_sar_hv = filt_mod_dfs[2]['s0_SAR_HV_(dB)'].to_numpy()
        cost_oh_hv = np.sum((s0_oh_hv - s0_sar_hv) ** 2)
        return cost_oh_hv
    # define the objective function for the Oh model (2002 version) - HV polarization

    def oh_vv_2002_obj_func(x):
        s0_oh_2002_vv = 10 * np.log10(0.11 * (x ** 0.7) * (np.cos(theta)) ** 2.2 * (1 - np.exp((-0.32) * (hrms_vals[3])
                                                                                               ** 1.8)) / (
                                              0.1 * ((khrms_vals[3] / l_corr[3]) + np.sin(1.3 * theta)) ** 1.2 *
                                              (1 - np.exp(-0.9 * khrms_vals[3]))))
        s0_sar_vv = filt_mod_dfs[3]['s0_SAR_VV_(dB)'].to_numpy()
        cost_oh_2002_vv = np.sum((s0_oh_2002_vv - s0_sar_vv) ** 2)
        return cost_oh_2002_vv
    # define the objective function for the Oh model (2002 version) - VV polarization

    def oh_hv_2004_obj_func(x):
        s0_oh_hv = 10 * np.log10(0.11 * (x ** 0.7) * (np.cos(theta)) ** 2.2 * (1 - np.exp((-0.32) * khrms_vals[4]
                                                                                          ** 1.8)))
        s0_sar_hv = filt_mod_dfs[4]['s0_SAR_HV_(dB)'].to_numpy()
        cost_oh_hv = np.sum((s0_oh_hv - s0_sar_hv) ** 2)
        return cost_oh_hv

    def oh_vv_2004_obj_func(x):
        s0_oh_2004_vv = 10 * np.log10(0.11 * (x ** 0.7) * (np.cos(theta)) ** 2.2 * (1 - np.exp((-0.32) * khrms_vals[5]
                                                                                               ** 1.8)) / (
                                              0.095 * (0.13 + np.sin(1.5 * theta)) ** 1.4 * (
                                              1 - np.exp(-1.3 * khrms_vals[5]
                                                         ** 0.9))))
        s0_sar_vv = filt_mod_dfs[5]['s0_SAR_VV_(dB)'].to_numpy()
        cost_oh_2004_vv = np.sum((s0_oh_2004_vv - s0_sar_vv) ** 2)
        return cost_oh_2004_vv

    # define the objective function for the Oh model (2004 version) - VV polarization

    def iem_fung_gauss_acf(x):
        invert_iem_dfs = comp_mod_dfs(SAR_s0_VV=None,
                                      SAR_s0_VH=None,
                                      l=5.5465763,
                                      theta=theta,
                                      sm=[x],
                                      dates=dates_list[6])
        # create an instance of the comp_mod_dfs() class from the "C1_backscatter_models_dataframes.py" script in order
        # to invert the Integral Equation Model (Fung Gaussian A.C.F.)

        s0_iem_fung_gauss_vv = invert_iem_dfs.iem_phys_model(hrms_vals=[hrms_vals[6]], st_sand_pct=st_sand_pct,
                                                             st_clay_pct=st_clay_pct, sar_freq=sar_freq,
                                                             l_corr_type='Taconet_Ciarletti', invert_mod=True,
                                                             inv_mod_type='Fung Gaussian')
        s0_sar_vv = filt_mod_dfs[6]['s0_SAR_VV_(dB)'].to_numpy()
        cost_iem_fung_gauss_acf = np.sum((s0_iem_fung_gauss_vv - s0_sar_vv) ** 2)
        return cost_iem_fung_gauss_acf

    # define the objective function for the Integral Equation Model (Fung Gaussian A.C.F.) - VV polarization

    def iem_brogioni_exp_acf(x):
        invert_iem_dfs = comp_mod_dfs(SAR_s0_VV=None,
                                      SAR_s0_VH=None,
                                      l=5.5465763,
                                      theta=theta,
                                      sm=[x],
                                      dates=dates_list[7])
        # create an instance of the comp_mod_dfs() class from the "C1_backscatter_models_dataframes.py" script in order
        # to invert the Integral Equation Model (Brogioni Exponential A.C.F.)

        s0_iem_brogioni_exp_vv = invert_iem_dfs.iem_phys_model(hrms_vals=[hrms_vals[7]],
                                                               st_sand_pct=st_sand_pct, st_clay_pct=st_clay_pct,
                                                               sar_freq=sar_freq, l_corr_type='Taconet_Ciarletti',
                                                               invert_mod=True, inv_mod_type='Brogioni Exponential')
        s0_sar_vv = filt_mod_dfs[7]['s0_SAR_VV_(dB)'].to_numpy()
        cost_iem_brogioni_exp_acf = np.sum((s0_sar_vv - s0_iem_brogioni_exp_vv) ** 2)
        return cost_iem_brogioni_exp_acf
    # define the objective function for the Integral Equation Model (Brogioni Exponential A.C.F.) - VV polarization

    def iem_b_fung_exp_acf(x):
        invert_iem_b_dfs = comp_mod_dfs(SAR_s0_VV=None,
                                        SAR_s0_VH=None,
                                        l=5.5465763,
                                        theta=theta,
                                        sm=[x],
                                        dates=dates_list[8])
        # create an instance of the comp_mod_dfs() class from the "C1_backscatter_models_dataframes.py" script in order
        # to invert the Integral Equation Model (Fung Exponential A.C.F.)

        s0_iem_b_fung_exp_vv = invert_iem_b_dfs.iem_phys_model(hrms_vals=[hrms_vals[8]],
                                                               st_sand_pct=st_sand_pct, st_clay_pct=st_clay_pct,
                                                               sar_freq=sar_freq, l_corr_type='Baghdadi',
                                                               invert_mod=True, inv_mod_type='Fung Exponential')
        s0_sar_vv = filt_mod_dfs[8]['s0_SAR_VV_(dB)'].to_numpy()
        cost_iem_b_fung_exp_acf = np.sum((s0_sar_vv - s0_iem_b_fung_exp_vv) ** 2)
        return cost_iem_b_fung_exp_acf

    # define the objective function for the Baghdadi Calibrated Integral Equation Model (Fung Gaussian A.C.F.) -
    # VV polarization

    def iem_b_brogioni_gauss_acf(x):
        invert_iem_dfs = comp_mod_dfs(SAR_s0_VV=None,
                                      SAR_s0_VH=None,
                                      l=5.5465763,
                                      theta=theta,
                                      sm=[x],
                                      dates=dates_list[9])
        # create an instance of the comp_mod_dfs() class from the "C1_backscatter_models_dataframes.py" script in order
        # to invert the Integral Equation Model (Gauss Exponential A.C.F.)

        s0_iem_brogioni_gauss_vv = invert_iem_dfs.iem_phys_model(hrms_vals=[hrms_vals[9]],
                                                                 st_sand_pct=st_sand_pct, st_clay_pct=st_clay_pct,
                                                                 sar_freq=sar_freq, l_corr_type='Baghdadi',
                                                                 invert_mod=True, inv_mod_type='Gauss Exponential')
        s0_sar_vv = filt_mod_dfs[9]['s0_SAR_VV_(dB)'].to_numpy()
        cost_iem_brogioni_gauss_acf = np.sum((s0_sar_vv - s0_iem_brogioni_gauss_vv) ** 2)
        return cost_iem_brogioni_gauss_acf

    # define the objective function for the Baghdadi Calibrated Integral Equation Model (Brogioni Gaussian A.C.F.) -
    # VV polarization

    obj_funcs_list = [nem_hv_obj_func, nem_vv_obj_func, oh_hv_2002_obj_func, oh_vv_2002_obj_func,
                      oh_hv_2004_obj_func, oh_vv_2004_obj_func, iem_fung_gauss_acf, iem_brogioni_exp_acf,
                      iem_b_fung_exp_acf, iem_b_brogioni_gauss_acf]
    mods_names = ('New Empirical Model - HV Polarization',
                  'New Empirical Model - VV Polarization', 'Oh Model 2002 - HV Polarization',
                  'Oh Model 2002 - VV Polarization', 'Oh Model 2004 - HV Polarization',
                  'Oh Model 2004 - VV Polarization', 'Integral Equation Model - Fung Gaussian A.C.F.',
                  'Integral Equation Model - Brogioni Exponential A.C.F.',
                  'Integral Equation Model (Baghdadi Calibrated)- Fung Exponential A.C.F.',
                  'Integral Equation Model (Baghdadi Calibrated)- Brogioni Gaussian A.C.F.')

    for i, obj_func in zip(range(len(obj_funcs_list)), obj_funcs_list):
        mod_moist_params = optimize.minimize(fun=obj_func, x0=sm[i], bounds=bounds_seqs[i], method='SLSQP',
                                             constraints=cons_per_mod[i], tol=1e-6)
        mod_moist_params = mod_moist_params.x

        a, b = np.polyfit(sm_per_mod[i], mod_moist_params, deg=1)
        mod_moist_params_func = a * sm_per_mod[i] + b
        rmse = np.sqrt(np.sum((mod_moist_params - mod_moist_params_func) ** 2 / len(mod_moist_params)))

        rss = np.sum((mod_moist_params - mod_moist_params_func) ** 2)
        tss = np.sum((mod_moist_params - np.mean(mod_moist_params)) ** 2)
        r_sq = 1 - rss / tss

        print(mod_moist_params)
        print(mods_names[i] + '\n' + 'RMSE=' + str(rmse) + '\n' + 'R^2=' + str(r_sq) + '\n\n\n')

        # fig, ax = plt.subplots()
        #
        # if 'dubois' in nameof(obj_func):
        #     ax.set_title('Regression Plot of In-Situ and Acquired Real Part of the Dielectric Constant \nfrom the '
        #                  'Inversion of the {} Objective Function'.format(mods_names[i]))
        #     ax.set_xlabel(r'$e_r$'+' Insitu (Dimensionless)')
        #     ax.set_ylabel(r'$e_r$'+' Inverted (Dimensionless)')
        #
        # else:
        #     ax.set_title('Regression Plot of In-Situ and Acquired Volumetric Soil Moisture \nfrom the '
        #                  'Inversion of the {} Objective Function'.format(mods_names[i]))
        #     ax.set_xlabel('mv Insitu ('+r'$m^3/m^3$'+')')
        #     ax.set_ylabel('mv Inverted ('+r'$m^3/m^3$'+')')
        #
        # ax.scatter(in_par_df['SM_(m^3/m^3)'], mod_moist_params.x_func)
        # ax.plot(in_par_df['SM_(m^3/m^3)'], a * in_par_df['SM_(m^3/m^3)'] + b, color='red')
        #
        # ax.text(0.8, 0.8, 'y=' + '{:.2f}'.format(a) + 'x+' + '{:.2f}'.format(b), transform=ax.transAxes,
        #         fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="cyan", ec="b", lw=2))
        # ax.text(0.8, 0.75, r'$R^2$'+'={:.2f}'.format(r_sq), transform=ax.transAxes, fontsize=12,
        #         bbox=dict(boxstyle="round,pad=0.3", fc="cyan", ec="b", lw=2))
        # ax.text(0.8, 0.7, 'RMSE={:.2f}'.format(rmse), transform=ax.transAxes, fontsize=12,
        #         bbox=dict(boxstyle="round,pad=0.3", fc="cyan", ec="b", lw=2))
        #
        # fig.save_fig()
