import pandas as pd
import numpy as np
from scipy.integrate import quad, dblquad
from B_prepare_data import model_variables


class comp_mod_dfs:
    """

    Class that contains functions which for 3 sets (empirical, semi-empirical and physical) of radar backscatter models
    computes the backscattering coefficient as a mathematical function of the soil moisture (or the dielectric constant),
    incidence angle, soil roughness etc. and for each date of the S1 images acquisition dates stores the backscatter
    coefficient values to a dataframe

    """

    def __init__(self, l, SAR_s0_VV, SAR_s0_VH, theta, sm, dates):
        """

        Args:
            SAR_s0_VH: the column of the initial parameters dataframe which contains the SAR GRD backscatter coefficient
            values for the VH polarization for each acquisition date
            SAR_s0_VV: the column of the initial parameters dataframe which contains the SAR GRD backscatter coefficient
            values for the VV polarization for each acquisition date
            l: the wavelength of the SAR signal
            theta: the column of the initial parameters dataframe which contains the incidence angle values for each
            S1 GRD data acquisition date (when creating an instance of the class for the inversion of the backscatter
            models a float number should be given instead of the column)
            sm: the column of the initial parameters dataframe which contains the volumetric soil moisture (m^3/m^3)
            values for each S1 GRD data acquisition date (please give the column as the only element of an 1-D list in
            order for the variable to be compatible e_sm_estimation() object of the class model_variables
            dates: the dates when the S1 GRD data was acquisitioned
        """

        self.SAR_s0_VV = SAR_s0_VV
        self.SAR_s0_VH = SAR_s0_VH
        self.l = l
        self.theta = theta
        self.sm = sm
        self.dates = dates

    def nem_mod(self, hrms_vals=np.linspace(0.01, 4.679, 6)):
        """
        Function that given certain parameters (soil moisture, incidence angle and root mean square error of the surface
        heights distribution) computes the modeled backscatter coefficient per the New Empirical model (Baghdadi et al.
        2015) in the VH and VV polarization

        Args:
            hrms_vals: a tuple of lists or numpy arrays (as many as the backscattering models used per function because
            each model has a different range of validity in regard to Hrms, if there is only one model use just a list
            or numpy array) which contain evenly spaced values of the root-mean-square error of the surface (where the
            ISMN station is located) height in a range of values (for ex. (np.linspace(0.1, 6, 6),
            np.linspace(0, 2.5, 6)) for 2 models which are valid in the ranges of Hrms [0.1, 6] and [0, 2.5] if one
            wants 6 Hrms values in the above-mentioned ranges

        Returns: a dataframe with the S1 acquired backscatter coefficients and the modeled backscatter coefficients for
        the New Empirical Model (HV and VV polarization)

        """
        k = (2 * np.pi) / self.l  # wave number of the SAR signal
        sm_per = self.sm[0] * 100  # volumetric soil moisture values dataframe column (%)

        empir_mods_dfs = []  # initialization of the list where the dataframes are going to be stored
        for i in range(len(hrms_vals)):
            empir_mods_dfs.append(pd.DataFrame(columns=["s0_SAR_HV_(dB)", "s0_SAR_VV_(dB)", "s0_N.E.M._VV_(dB)",
                                                        "s0_N.E.M._HV_(dB)", "s0_SAR_HV_NaN_Vals", "s0_SAR_VV_NaN_Vals"]
                                               , index=self.dates))
            # definition of the columns and the index for each dataframe in the list

        for j in range(len(hrms_vals)):
            empir_mods_dfs[j]['s0_SAR_HV_(dB)'] = self.SAR_s0_VH
            empir_mods_dfs[j]['s0_SAR_VV_(dB)'] = self.SAR_s0_VV
            # modification of the variables contained in the initial parameters dataframes and passing of the S1
            # acquired mean backscatter coefficients

            empir_mods_dfs[j]['s0_SAR_HV_NaN_Vals'] = empir_mods_dfs[j]['s0_SAR_HV_(dB)'].isnull()
            empir_mods_dfs[j]['s0_SAR_VV_NaN_Vals'] = empir_mods_dfs[j]['s0_SAR_VV_(dB)'].isnull()
            # create a column with boolean values indicating if a SAR backscatter coefficient is null or not

            for h in range(len(sm_per)):
                if not empir_mods_dfs[j]['s0_SAR_VV_NaN_Vals'][h]:
                    empir_mods_dfs[j].iloc[h, empir_mods_dfs[j].columns.get_loc('s0_N.E.M._VV_(dB)')] = 10 ** -1.287 * \
                                          (np.cos(self.theta[h])) ** 1.227 * 10 ** (0.009 * (1 / np.tan(self.theta[h]))
                                        * sm_per[h]) * (k * hrms_vals[j]) ** (0.86 * np.sin(self.theta[h]))
                    empir_mods_dfs[j].iloc[h, empir_mods_dfs[j].columns.get_loc('s0_N.E.M._VV_(dB)')] = \
                        10 * np.log10(empir_mods_dfs[j]['s0_N.E.M._VV_(dB)'][h])
                else:
                    empir_mods_dfs[j].iloc[h, empir_mods_dfs[j].columns.get_loc('s0_N.E.M._VV_(dB)')] = np.nan

                if not empir_mods_dfs[j]['s0_SAR_HV_NaN_Vals'][h]:
                    empir_mods_dfs[j].iloc[h, empir_mods_dfs[j].columns.get_loc('s0_N.E.M._HV_(dB)')] = \
                        10 ** (-2.325) * (np.cos(self.theta[h])) ** (-0.01) * 10 ** (0.011 * (1 / np.tan(self.theta[h]))
                      * sm_per[h]) * (k * hrms_vals[j]) ** (0.44 * np.sin(self.theta[h]))
                    empir_mods_dfs[j].iloc[h, empir_mods_dfs[j].columns.get_loc('s0_N.E.M._HV_(dB)')] = \
                        10 * np.log10(empir_mods_dfs[j]['s0_N.E.M._HV_(dB)'][h])
                else:
                    empir_mods_dfs[j].iloc[h, empir_mods_dfs[j].columns.get_loc('s0_N.E.M._HV_(dB)')] = np.nan
                    # computation of the backscatter coefficients (dB) from the New Empirical Model for the HV and
                    # VV polarization, for each date and for each roughness index value

            empir_mods_dfs[j].drop(axis=1, columns=['s0_SAR_HV_NaN_Vals', 's0_SAR_VV_NaN_Vals'], inplace=True)
            # drop the columns containing boolean values indicating if a SAR backscatter coefficient is null or not

        return empir_mods_dfs

    def oh_mod_vers(self, hrms_vals=(np.linspace(0.088, 6.179, 6), np.linspace(0.01, 6.16, 6))):
        """
        Function that given certain parameters (soil moisture, incidence angle and root mean square error of the surface
        heights distribution) computes the modeled backscatter coefficient per the Oh model (2002 and 2004 versions) in
        the VH and VV polarization

        Args:
            hrms_vals: same type as in the other functions

        Returns: a dataframe with the S1 acquired backscatter coefficients and the modeled backscatter coefficients for
        the Oh model versions from 2002 and 2004 (HV and VV polarization)

        """
        k = (2 * np.pi) / self.l  # wave number of the SAR signal
        sm_vol = self.sm[0]  # volumetric soil moisture values dataframe column (m^3/m^3)
        l_corr = [hrms_vals[0][j] ** 2 * (1 / np.e) for j in range(len(hrms_vals[0]))]
        # generation of lists for each Hrms list and each Oh model version where the correlation length is computed as
        # formulated by Taconet and Ciarletti 2007

        oh_mods_dfs = []  # initialization of the list where the dataframes are going to be stored
        for i in range(len(hrms_vals[0])):
            oh_mods_dfs.append(pd.DataFrame(columns=['s0_SAR_HV_(dB)', 's0_SAR_VV_(dB)', 's0_Oh-2002_HV_(dB)',
                                                     's0_Oh-2004_HV_(dB)', 's0_Oh-2002_VV_(dB)', 's0_Oh-2004_VV_(dB)'],
                                            index=self.dates))  # definition of the columns and the index for
            # each dataframe in the list
            oh_mods_dfs[i]['s0_SAR_HV_(dB)'] = self.SAR_s0_VH
            oh_mods_dfs[i]['s0_SAR_VV_(dB)'] = self.SAR_s0_VV

            oh_mods_dfs[i]['s0_SAR_HV_NaN_Vals'] = oh_mods_dfs[i]['s0_SAR_HV_(dB)'].isnull()
            oh_mods_dfs[i]['s0_SAR_VV_NaN_Vals'] = oh_mods_dfs[i]['s0_SAR_VV_(dB)'].isnull()
            # create a column with boolean values indicating if a SAR backscatter coefficient is null or not

        for j in range(len(hrms_vals[0])):
            for h in range((len(sm_vol))):
                if sm_vol[h] > 0.03 and not oh_mods_dfs[j]['s0_SAR_HV_NaN_Vals'][h] and not \
                        oh_mods_dfs[j]['s0_SAR_VV_NaN_Vals'][h]:
                    oh_mods_dfs[j].iloc[h, oh_mods_dfs[j].columns.get_loc('s0_Oh-2002_HV_(dB)')] = \
                        0.11 * (sm_vol[h] ** 0.7) * (np.cos(self.theta[h])) ** 2.2 * (1 - np.e ** (-0.32 * (k *
                        hrms_vals[0][j])))
                    oh_mods_dfs[j].iloc[h, oh_mods_dfs[j].columns.get_loc('s0_Oh-2002_VV_(dB)')] = \
                        oh_mods_dfs[j].iloc[h, oh_mods_dfs[j].columns.get_loc('s0_Oh-2002_HV_(dB)')] / \
                        (0.10 * (hrms_vals[0][j] / l_corr[j]) + np.sin(1.3 * self.theta[h])) ** 1.2 * \
                        (1 - np.e ** (-0.9 * (k * hrms_vals[0][j]) ** 0.8))
                    oh_mods_dfs[j].iloc[h, oh_mods_dfs[j].columns.get_loc('s0_Oh-2002_HV_(dB)')] \
                        = 10 * np.log10(oh_mods_dfs[j].iloc[h, oh_mods_dfs[j].columns.get_loc('s0_Oh-2002_HV_(dB)')])
                    oh_mods_dfs[j].iloc[h, oh_mods_dfs[j].columns.get_loc('s0_Oh-2002_VV_(dB)')] \
                        = 10 * np.log10(oh_mods_dfs[j].iloc[h, oh_mods_dfs[j].columns.get_loc('s0_Oh-2002_VV_(dB)')])
                    # computation of the backscatter coefficient in the HV polarization according to the Oh model for
                    # each Hrms value
                else:
                    oh_mods_dfs[j].iloc[h, oh_mods_dfs[j].columns.get_loc('s0_Oh-2002_HV_(dB)')] = np.nan
                    oh_mods_dfs[j].iloc[h, oh_mods_dfs[j].columns.get_loc('s0_Oh-2002_VV_(dB)')] = np.nan

                if sm_vol[h] > 0.04 and not oh_mods_dfs[j]['s0_SAR_HV_NaN_Vals'][h] \
                        and not oh_mods_dfs[j]['s0_SAR_VV_NaN_Vals'][h]:
                    oh_mods_dfs[j].iloc[h, oh_mods_dfs[j].columns.get_loc('s0_Oh-2004_HV_(dB)')] = \
                        0.11 * (sm_vol[h] ** 0.7) * (np.cos(self.theta[h])) ** 2.2 * (1 - np.e ** (-0.32 * (k *
                        hrms_vals[1][j])))
                    oh_mods_dfs[j].iloc[h, oh_mods_dfs[j].columns.get_loc('s0_Oh-2004_VV_(dB)')] = \
                        oh_mods_dfs[j].iloc[h, oh_mods_dfs[j].columns.get_loc('s0_Oh-2004_HV_(dB)')] / 0.095 * (
                                0.13 + np.sin(1.5 * self.theta[h])) ** 1.4 * (
                                1 - np.e ** (- 1.3 * (k * hrms_vals[1][j]) ** 0.9))
                    oh_mods_dfs[j].iloc[h, oh_mods_dfs[j].columns.get_loc('s0_Oh-2004_HV_(dB)')] = \
                        10 * np.log10(oh_mods_dfs[j].iloc[h, oh_mods_dfs[j].columns.get_loc('s0_Oh-2004_HV_(dB)')])
                    oh_mods_dfs[j].iloc[h, oh_mods_dfs[j].columns.get_loc('s0_Oh-2004_VV_(dB)')] = \
                        10 * np.log10(oh_mods_dfs[j].iloc[h, oh_mods_dfs[j].columns.get_loc('s0_Oh-2004_VV_(dB)')])
                    # computation of the backscatter coefficient according to the Oh model 2004 (correlation length is
                    # not accounted for due to it not being able to be measured accurately)
                else:
                    oh_mods_dfs[j].iloc[h, oh_mods_dfs[j].columns.get_loc('s0_Oh-2004_HV_(dB)')] = np.nan
                    oh_mods_dfs[j].iloc[h, oh_mods_dfs[j].columns.get_loc('s0_Oh-2004_VV_(dB)')] = np.nan
                    # if soil moisture values given are very small replace the cells of the dataframe with nan values

            oh_mods_dfs[j].drop(axis=1, columns=['s0_SAR_HV_NaN_Vals', 's0_SAR_VV_NaN_Vals'], inplace=True)
            # drop the columns containing boolean values indicating if a SAR backscatter coefficient is null or not

        return oh_mods_dfs

    def dubois_mod(self, hrms_vals=np.linspace(0.01, 2.207, 6), st_clay_pct=None, st_sand_pct=None, sar_freq=None,
                   invert_mod=False):
        """
                Function that given certain parameters (real part of the complex dielectric constant derived from the
                soil moisture with the help of the Hallikainen model (Hallikainen et al. 2015), incidence angle and root
                mean square error of the surface heights distribution) computes the modeled backscatter coefficient per
                the Dubois model (Dubois et al. 1995) in the VV polarization

                Args:
                    hrms_vals: same as in the above functions (when using the function for inverting the model give a
                    single value representing the Hrms value for which the optimal results are achieved
                    st_clay_pct: the clay percentage of the soil in the I.S.M.N. station
                    st_sand_pct: the sand percentage of the soil in the I.S.M.N. station
                    sar_freq: the frequency of the SAR radar system used in GHz
                    invert_mod: a boolean indicating if the function is used for the computation of the backscatter
                    coefficient values or the inversion of the model

                Returns: a dataframe with the S1 acquired backscatter coefficients and the modeled backscatter
                coefficients for the Dubois Model (VV polarization)

                """
        k = (2 * np.pi) / self.l  # wave number of the SAR signal
        sm_vol = self.sm[0]  # volumetric soil moisture values dataframe column (m^3/m^3)

        if not invert_mod:
            comp_er_vars = model_variables(date_range=self.dates,
                                           image_list=None,
                                           sm_datafile=self.sm,
                                           s0_timeseries=None,
                                           clay_pct=st_clay_pct,
                                           sand_pct=st_sand_pct,
                                           freq_GHZ=sar_freq)
        else:
            comp_er_vars = model_variables(date_range=self.dates,
                                           image_list=None,
                                           sm_datafile=[pd.Series(data=sm_vol, index=self.dates, name='SM_(m^3/m^3)')],
                                           s0_timeseries=None,
                                           clay_pct=st_clay_pct,
                                           sand_pct=st_sand_pct,
                                           freq_GHZ=sar_freq)

        dubois_er_df = comp_er_vars.e_estimation(ismn_data=False)
        dubois_er_df.drop(columns=['e_imag_(Hallikainen)'], inplace=True, axis=1)
        e_r = dubois_er_df['e_real_(Hallikainen)']
        # create an instance of the model_variables class and generate the dataframe containing the real part of the
        # complex dielectric constant

        if not invert_mod:
            dubois_mod_dfs = []  # initialization of the list where the dataframes are going to be stored
            for i in range(len(hrms_vals)):
                dubois_mod_dfs.append(pd.DataFrame(columns=["s0_SAR_VV_(dB)", "s0_Dubois_VV_(dB)", "s0_SAR_VV_NaN_Vals"]
                                                   , index=self.dates))
                # definition of the columns and the index for each dataframe in the list

            for j in range(len(hrms_vals)):
                dubois_mod_dfs[j]['s0_SAR_VV_(dB)'] = self.SAR_s0_VV
                # modification of the variables contained in the initial parameters dataframes and passing of the S1
                # acquired mean backscatter coefficients

                dubois_mod_dfs[j]['s0_SAR_VV_NaN_Vals'] = dubois_mod_dfs[j]['s0_SAR_VV_(dB)'].isnull()
                # create a column with boolean values indicating if a SAR backscatter coefficient is null or not

                for h in range(len(sm_vol)):
                    if not dubois_mod_dfs[j]['s0_SAR_VV_NaN_Vals'][h]:
                        dubois_mod_dfs[j].iloc[h, dubois_mod_dfs[j].columns.get_loc('s0_Dubois_VV_(dB)')] = \
                            10 ** (-2.35) * (np.cos(self.theta[h]) ** 3 / np.sin(self.theta[h])) * 10 ** (0.046 * e_r[h] *
                            np.tan(self.theta[h])) * (k * hrms_vals[j] * np.sin(self.theta[h]) ** 3) ** 1.1 * self.l ** 0.7
                        dubois_mod_dfs[j].iloc[h, dubois_mod_dfs[j].columns.get_loc('s0_Dubois_VV_(dB)')] = \
                            10 * np.log10(dubois_mod_dfs[j]['s0_Dubois_VV_(dB)'][h])
                    else:
                        dubois_mod_dfs[j].iloc[h, dubois_mod_dfs[j].columns.get_loc('s0_Dubois_VV_(dB)')] = np.nan
                        # computation of the backscatter coefficients (dB) from the Dubois Model for the VV polarization,
                        # for each date and for each roughness index value

                dubois_mod_dfs[j].drop(axis=1, columns=['s0_SAR_VV_NaN_Vals'], inplace=True)
                # drop the columns containing boolean values indicating if a SAR backscatter coefficient is null or not
            return dubois_mod_dfs
        else:
            dubois_expr = 10 * np.log10((10 ** -2.35) * (np.cos(self.theta) ** 3 / np.sin(self.theta)) * 10 **
                                        (0.046 * e_r * np.cos(self.theta)) * (k * hrms_vals *
                                        np.sin(self.theta) ** 3) ** 1.1 * self.l ** 0.7)
            # construct the mathematical expression of the Dubois model needed for the inversion of the model
            return dubois_expr

    def iem_phys_model(self, hrms_vals=np.linspace(0.088, 2.648, 6), st_clay_pct=None, st_sand_pct=None, sar_freq=None,
                       l_corr_type=None, invert_mod=False, inv_mod_type=None):
        """

        Args:
            hrms_vals: same as in the above function (if the function is used for model inversion give the optimal Hrms
            value as a list of one value)
            l_corr_type: a string which identifies the type of the correlation length function
            st_clay_pct: same as in the above function
            st_sand_pct: same as in the above function
            sar_freq: same as in the above function
            invert_mod: a boolean indicating if the function is used for the computation of the backscatter
            coefficient values or the inversion of the mode
            inv_mod_type: the type of the A.C.F. used in the model

        Returns: the dataframe which contains the modeled backscatter coefficient values for the VV polarization and in
        accordance with the correlation function and the autocorrelation function type

        """
        k = (2 * np.pi) / self.l  # wave number of the SAR signal
        sm_vol = self.sm[0]  # volumetric soil moisture values dataframe column (m^3/m^3)
        # first intialize the parameters needed for the estimation of the VV polarized backscatter coefficients
        # according to the Simplified Integral Equation Model [Fung 2010]

        if not invert_mod:
            comp_er_vars = model_variables(date_range=self.dates,
                                           image_list=None,
                                           sm_datafile=self.sm,
                                           s0_timeseries=None,
                                           clay_pct=st_clay_pct,
                                           sand_pct=st_sand_pct,
                                           freq_GHZ=sar_freq)
        else:
            comp_er_vars = model_variables(date_range=self.dates,
                                           image_list=None,
                                           sm_datafile=[pd.Series(data=sm_vol, index=self.dates, name='SM_(m^3/m^3)')],
                                           s0_timeseries=None,
                                           clay_pct=st_clay_pct,
                                           sand_pct=st_sand_pct,
                                           freq_GHZ=sar_freq)

        iem_e_df = comp_er_vars.e_estimation(ismn_data=False)
        e_compl = iem_e_df['e_real_(Hallikainen)'] + 1j * iem_e_df['e_imag_(Hallikainen)']
        # create an instance of the model_variables class and generate the dataframe containing the complex dielectric
        # constant

        iem_mod_dfs = []  # initialization of the list where the dataframes are going to be stored
        for i in range(len(hrms_vals)):
            l_corr = None
            if l_corr_type == 'Taconet_Ciarletti':
                l_corr = hrms_vals[i] ** 2 * (1 / np.e)
            elif l_corr_type == 'Baghdadi':
                l_corr = 1.281 + 0.134 * (np.sin(0.19 * self.theta)) ** (-1.59) * hrms_vals[i]
                # compute the correlation length for each Hrms value
            iem_mod_dfs.append(pd.DataFrame(columns=['s0_SAR_VV_(dB)', 's0_IEM_VV_(dB)_Gauss_ACF_Fung',
                                                     's0_IEM_VV_(dB)_Exp_ACF_Fung', 's0_IEM_VV_(dB)_Gauss_ACF_Brogioni',
                                                     's0_IEM_VV_(dB)_Exp_ACF_Brogioni'], index=self.dates))
            # definition of the columns and the index for each dataframe in the list

            iem_mod_dfs[i]['s0_SAR_VV_(dB)'] = self.SAR_s0_VV
            iem_mod_dfs[i]['s0_SAR_VV_NaN_Vals'] = iem_mod_dfs[i]['s0_SAR_VV_(dB)'].isnull()
            # create a column with boolean values indicating if a SAR backscatter coefficient is null or not

            Rv = (-e_compl * np.cos(self.theta) + np.sqrt(e_compl - np.sin(self.theta) ** 2)) / \
                 (e_compl * np.cos(self.theta) + np.sqrt(e_compl - np.sin(self.theta) ** 2))
            Rv0 = (-e_compl * np.cos(0.) + np.sqrt(e_compl - np.sin(0.) ** 2)) / \
                  (e_compl * np.cos(0.) + np.sqrt(e_compl - np.sin(0.) ** 2))
            # compute the complex fresnel reflectivity coefficient defined as a function of the complex dielectric
            # constant of the illuminated target by the SAR radar for the incidence angle of the radar beam and the
            # at nadir

            Ft = 8 * Rv0 ** 2 * np.sin(self.theta) ** 2 * ((np.cos(self.theta + np.sqrt(e_compl - np.sin(self.theta) ** 2)))
                                                      / (np.cos(self.theta)) * np.sqrt(e_compl - np.sin(self.theta) ** 2))
            St0 = 1 / (1 + (8 * Rv0) / (Ft * np.cos(self.theta))) ** 2

            def field_coeffs(acf_type):
                """

                Args:
                    acf_type: a string identifier for the type of the ACF function

                Returns: the numpy array with the sigma0 backscatter coefficient values for each ACF and Hrms value in
                the validity range

                """
                acf = None
                n = 1
                assert acf_type in ['Brogioni_Gauss', 'Brogioni_Exp', 'Fung_Gauss', 'Fung_Exp']

                if acf_type == 'Fung_Exp':
                    acf = (2 * np.pi * n * l_corr ** 2) / (n ** 2 + (2 * k * l_corr * np.sin(self.theta))) ** 1.5
                elif acf_type == 'Fung_Gauss':
                    acf = ((np.pi * l_corr) / n) * np.exp((-k * l_corr * np.sin(self.theta)) ** 2 / n)
                elif acf_type == 'Brogioni_Gauss':
                    acf = 2 * np.pi * (l_corr / 2 * n) ** 2 * np.exp((1 + k ** 2 * l_corr ** 2) / n ** 2)
                elif acf_type == 'Brogioni_Exp':
                    acf = (l_corr ** 2 / (2 * n)) * np.exp((-k * l_corr) / 4 * n)
                    # define a function returning the two autocorrelation functions (Gaussian and Exponential) as they
                    # were defined by Fung 2010 and Brogioni et al. 2010 for each Hrms value

                threshold = 1e-8

                sum_numerator = np.float64(0)
                converged = False
                while not converged:
                    sum_numer_iter = (k * hrms_vals[i] * np.cos(self.theta)) ** (2 * n) / np.math.factorial(n) \
                                     * acf
                    sum_numerator += sum_numer_iter.astype(float)
                    n += 1
                    error = np.abs(sum_numer_iter / sum_numerator)
                    if np.all(error < threshold):
                        converged = True

                sum_denominator = np.float64(0)
                n = 1
                converged = False
                while not converged:
                    term = (k * hrms_vals[i] * np.cos(self.theta)) ** (2 * n) / np.math.factorial(n) \
                           * np.abs(Ft + 2 ** (n + 2) * Rv0 / (np.exp((k * hrms_vals[i] * np.cos(self.theta)) ** 2) *
                                                               np.cos(self.theta))) ** 2 * acf
                    sum_denominator += term.astype(float)
                    n += 1
                    error = np.abs(term / sum_denominator)
                    if np.all(error < threshold):
                        converged = True

                St = np.abs(Ft) ** 2 * sum_numerator / sum_denominator
                Rvt = Rv + (Rv0 - Rv) * (1 - St / St0)
                # compute the generalized fresnel coefficient for the VV polarization as reflection transition functions
                # (defined by Fung 2010) for each Hrms value and surface spectra

                sq = np.sqrt(e_compl - np.sin(self.theta) ** 2)
                Tvm = 1 - Rvt
                Tv = 1 + Rvt
                fvv = (2 * Rvt) / np.cos(self.theta)
                Fvv = ((np.sin(self.theta) ** 2 / np.cos(self.theta) ** 2) - (sq / e_compl)) * Tv ** 2 - \
                      (2 * np.sin(self.theta) ** 2) * ((1 / np.cos(self.theta)) + (1 / sq)) * Tv * Tvm + \
                      ((np.sin(self.theta) ** 2 / np.cos(self.theta)) + e_compl * (1 + np.sin(self.theta) ** 2) / sq) * \
                      Tvm ** 2
                # compute the field coefficients for the VV polarization (fresnel transition functions)

                summation = 0
                n = 1
                converged = False
                while not converged:
                    I_ppn = (2 * k * hrms_vals[i] * np.cos(self.theta)) ** n * fvv * np.exp(-(k * hrms_vals[i] *
                             np.cos(self.theta) ** 2)) + (k * hrms_vals[i] * np.cos(self.theta)) ** n * Fvv
                    term = np.abs(I_ppn) ** 2 * acf / np.math.factorial(n)
                    summation += np.float64(term)
                    n += 1
                    error = np.abs(term / summation)
                    if np.all(error < threshold):
                        converged = True

                sigma = k ** 2 / (4 * np.pi) * np.exp(-2 * (k * hrms_vals[i] * np.cos(self.theta) ** 2)) * summation
                sigma_dB = 10 * np.log10(sigma)
                # finally, compute the backscatter coefficient numpy arrays for the I.E.M. model and each hrms value
                # in the validity range of the model
                return sigma_dB

            sigma_dB_Gauss_Fung = field_coeffs(acf_type='Fung_Gauss')
            sigma_dB_Exp_Fung = field_coeffs(acf_type='Fung_Exp')
            sigma_dB_Gauss_Brogioni = field_coeffs(acf_type='Brogioni_Gauss')
            sigma_dB_Exp_Brogioni = field_coeffs(acf_type='Brogioni_Exp')

            if inv_mod_type == 'Fung Gaussian':
                return sigma_dB_Gauss_Fung
            elif inv_mod_type == 'Fung Exponential':
                return sigma_dB_Exp_Fung
            elif inv_mod_type == 'Brogioni Gaussian':
                return sigma_dB_Gauss_Brogioni
            elif inv_mod_type == 'Brogioni Exponential':
                return sigma_dB_Exp_Brogioni
            else:
                pass

            for h in range((len(sm_vol))):
                if not iem_mod_dfs[i]['s0_SAR_VV_NaN_Vals'][h]:
                    iem_mod_dfs[i].iloc[h, iem_mod_dfs[i].columns.get_loc('s0_IEM_VV_(dB)_Gauss_ACF_Fung')] = \
                        sigma_dB_Gauss_Fung[h]
                    iem_mod_dfs[i].iloc[h, iem_mod_dfs[i].columns.get_loc('s0_IEM_VV_(dB)_Gauss_ACF_Brogioni')] = \
                        sigma_dB_Gauss_Brogioni[h]
                    iem_mod_dfs[i].iloc[h, iem_mod_dfs[i].columns.get_loc('s0_IEM_VV_(dB)_Exp_ACF_Fung')] = \
                        sigma_dB_Exp_Fung[h]
                    iem_mod_dfs[i].iloc[h, iem_mod_dfs[i].columns.get_loc('s0_IEM_VV_(dB)_Exp_ACF_Brogioni')] = \
                        sigma_dB_Exp_Brogioni[h]
                else:
                    iem_mod_dfs[i].iloc[h, iem_mod_dfs[i].columns.get_loc('s0_IEM_VV_(dB)_Gauss_ACF_Fung')] = np.nan
                    iem_mod_dfs[i].iloc[h, iem_mod_dfs[i].columns.get_loc('s0_IEM_VV_(dB)_Gauss_ACF_Brogioni')] = \
                        np.nan
                    iem_mod_dfs[i].iloc[h, iem_mod_dfs[i].columns.get_loc('s0_IEM_VV_(dB)_Exp_ACF_Fung')] = np.nan
                    iem_mod_dfs[i].iloc[h, iem_mod_dfs[i].columns.get_loc('s0_IEM_VV_(dB)_Exp_ACF_Brogioni')] = np.nan

            iem_mod_dfs[i].drop(axis=1, columns=['s0_SAR_VV_NaN_Vals'], inplace=True)
            # drop the columns containing boolean values indicating if a SAR backscatter coefficient is null or not

        return iem_mod_dfs
