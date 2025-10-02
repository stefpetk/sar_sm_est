#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# MODULES AND FUNCTIONS NEEDED FOR THE PROGRAM #

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit


class model_analysis:
    """
    Class which includes functions needed for the comparison of the SAR acquired and modeled backscatter coefficients
    and the time series of the SAR acquired and modeled backscatter coefficients
    """

    def __init__(self, date_range, model_dfs, wavelength, model_list, pol_list, station):
        """

        Args:
            date_range: a list with the acquisition dates of the S1 images
            hrms_list: a tuple of lists or numpy arrays (as many as the backscattering models used per function because
            each model has a different range of validity in regard to Hrms, if there is only one model use just a list
            or numpy array) which contain evenly spaced values of the root-mean-square error of the surface (where the
            ISMN station is located) height in a range of values (for ex. (np.linspace(0.1, 6, 6),
            np.linspace(0, 2.5, 6)) for 2 models which are valid in the ranges of Hrms [0.1, 6] and [0, 2.5] if one
            wants 6 Hrms values in the above mentioned ranges)
            in_par_df: the dataframe which contains the SAR acquired backscatter coefficients and the ISMN data
            model_dfs: the dataframe which contains the SAR acquired and modeled backscatter coefficients
            wavelength: a float number that contains the value of the central wavelength of the SAR
            model_list: a list of strings that contains the backscatter model
            pol_list: a list of strings that contains the polarization of the modeled or SAR acquired backscatter
            coefficient
            station: a string which acts as an identifier for the ISMN station
        """
        self.date_range = date_range
        self.model_dfs = model_dfs
        self.wavelength = wavelength
        self.model_list = model_list
        self.pol_list = pol_list
        self.station = station

    def forward_modelling(self):

        for model in  self.model_list:
            for pol in self.pol_list:
                for j in range(len(self.hrms_list[i])):
                    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 3))
                    ax.set_title("$σ_0({})$".format(pol) + " - ({} Model) ".format(model) +
                                 "$σ_0({})$ (SAR) Scatter Plot for kHrms=".format(pol) +
                                 '{:.2f}'.format((2 * np.pi / self.wavelength) * self.hrms_list[i][j]))
                    ax.set_xlabel(r"$σ_0({})$".format(pol) + " - SAR (dB)")
                    ax.set_ylabel(r"$σ_0({})$".format(pol) + " - {} (dB)".format(model))
                    meas_s0_vals = 's0_SAR_{}_(dB)'.format(pol)
                    mod_s0_vals = 's0_{}'.format(model) + '_{}_(dB)'.format(pol)
                    if mod_s0_vals != 's0_Dubois_HV_(dB)' and self.model_dfs[j][mod_s0_vals].all():
                        ax.scatter(self.in_par_df[meas_s0_vals],
                                   self.model_dfs[j][mod_s0_vals], color='blue')
                        # if mod_s0_vals in ['s0_Oh_2002_HV_(dB)', 's0_Oh_2004_HV_(dB)', 's0_Oh_2002_VV_(dB)',
                        #                    's0_Oh_2004_VV_(dB)']:
                        #     a, b = np.polyfit(self.model_dfs[j][meas_s0_vals], self.model_dfs[j][mod_s0_vals], deg=1)
                        # else:
                        a, b = np.polyfit(self.in_par_df[meas_s0_vals], self.model_dfs[j][mod_s0_vals], deg=1)
                        ax.plot(self.in_par_df[meas_s0_vals], a * self.in_par_df[meas_s0_vals] + b, color='red')

                        ydat_mean = np.mean(self.model_dfs[j][mod_s0_vals])
                        res_sum_sq = np.sum(
                            (self.model_dfs[j][mod_s0_vals] - (a * self.in_par_df[meas_s0_vals] + b)) ** 2)
                        tot_sum_sq = np.sum((self.model_dfs[j][mod_s0_vals] - ydat_mean) ** 2)
                        r_sq = 1 - res_sum_sq / tot_sum_sq
                        rmse = np.sqrt(np.sum((((a * self.in_par_df[meas_s0_vals] + b) -
                                                self.model_dfs[j][mod_s0_vals]) ** 2) /
                                              len(self.model_dfs[j][mod_s0_vals])))

                        ax.text(0.1, 0.9, '$R^2$=' + '{:.2f}'.format(r_sq), fontsize=14, transform=ax.transAxes)
                        ax.text(0.1, 0.8, 'RMSE=' + '{:.2f}'.format(rmse) + ' dB', fontsize=14, transform=ax.transAxes)
                        plt.savefig('C:/Users/stef-/Desktop/Thesis/Data-Proccessing/Results'
                                    '/SM_Intensity_Forward_Modelling_Curve_Fitting/Temp_Results/{}/'.format(self.station) +
                                    'scatter_plot_{}'.format(model) + '_model_{}'.format(pol) +
                                    '_khrms={}'.format(self.hrms_list[i][j]) + '.png', bbox_inches='tight',
                                    orientation='landscape', dpi=400, papertype='a3')
                        plt.show()

    #             def best_fit_curve(x, a, b, c):
    #                 return a * (x - b) ** 2 + c
    #
    #             min_mod_dfs_index = int(np.where(self.model_dfs[i][mod_val_string] ==
    #                                              min(self.model_dfs[i][mod_val_string]))[0])
    #             popt, pcov = curve_fit(best_fit_curve, self.in_par_df[meas_val_string],
    #                                    self.model_dfs[i][mod_val_string], p0=(0.03, max(self.in_par_df[meas_val_string]),
    #                                                                           min(self.model_dfs[i][mod_val_string])))
    #             a_opt, b_opt, c_opt = popt
    #             y_model = best_fit_curve(np.sort(self.in_par_df[meas_val_string]), a_opt, b_opt, c_opt)
    #             # creation of the scatter plot and the best fit curve computed through least squares
    #
    #             y_mean.append(np.mean(self.model_dfs[i][mod_val_string]))
    #             y_pred.append(a * self.in_par_df[meas_val_string] + b)
    #             tot_sum_sq.append(np.sum((self.model_dfs[i][mod_val_string] - y_mean[i]) ** 2))
    #             res_sum_sq.append(np.sum((self.model_dfs[i][mod_val_string] - y_pred[i]) ** 2))
    #             r_sq.append(1 - res_sum_sq[i] / tot_sum_sq[i])
    #             rmse.append(np.sum(y_pred[i] - self.model_dfs[i][mod_val_string]) / len(self.date_range))
    #             # append the before mentioned lists with the values of the mean of the dependent variable, the predicted
    #             # value of the dependent variable, the total sum of squares, the residual sum of squares #
    #             # and the R squared and also compute the R.M.S.E. (Root Mean Square Error) for the best fit curve #
    #
    #             ax1.plot(self.in_par_df[meas_val_string], a * self.in_par_df[meas_val_string] + b,
    #                      color='red', linestyle='-', linewidth=2, ms=5)
    #             ax2.plot(np.sort(self.in_par_df[meas_val_string]), y_model, color='green')  # plot the best fit
    #             # line and curve
    #             ax1.text(0.75, 0.8, 'y= ' + '{:.2f}'.format(b) + '{:.2f}'.format(a) + 'x', size=15,
    #                      transform=ax1.transAxes)
    #             ax2.text(0.75, 0.8, 'y= ' + '{:.2f}'.format(a_opt) + '$(x{:.2f})^2$'.format(b_opt) + '{:.2f}'.
    #                      format(c_opt), size=15, transform=ax2.transAxes)
    #             ax1.text(0.75, 0.75, '$R^2$={:.2f}'.format(r_sq[i]), size=15, transform=ax1.transAxes)
    #             ax2.text(0.75, 0.75, 'RMSE={:.2f}'.format(rmse[i]), size=15, transform=ax2.transAxes)
    #             plt.show()
    #             # plt.savefig('regression_plots_{}_'.format(self.model) + '_{}_'.format(self.pol) + 'khrms_' +
    #             #             '{:.2f}'.format(self.wavelength * self.hrms_list[i]) + ' {} model'.format(
    #             #     self.e_est_model) + '.png', dpi=300.0)
    #             # save the regression plots for each k*Hrms value
    #
    # # FUNCTION WHICH PLOTS THE TIME SERIES OF THE MODELED BACKSCATTER COEFFICIENTS AND THE ONES ACQUIRED #
    # # BY THE SAR IMAGES #

    def time_series(self):
        meas_val_string = 's0_SAR_{}_(dB)'.format(self.pol)
        mod_val_string = 's0_{}'.format(self.model) + '_{} (dB)'.format(self.pol)
        # string variables for accessing the correct #
        # dataframes according to the backscatter model and the polarization or the Oh model version #

        for i in range(len(self.hrms_list)):
            plt.plot(self.date_range, self.in_par_df[meas_val_string])
            plt.plot(self.date_range, self.model_dfs[i][mod_val_string])
            self.in_par_df[meas_val_string].plot()
            self.model_dfs[i][mod_val_string].plot()
            plt.title("$σ^0_{}$".format(self.pol) + " ({})".format(
                self.model) + "- $σ^0_{}$ (SAR) Time Series for kHrms=".format(self.pol)
                      + '{:.2f}'.format((2 * np.pi / self.wavelength) * self.hrms_list[i]))
            plt.ylabel("$σ^0_{}$ (dB)".format(self.pol))
            plt.xlabel("DOY")
            plt.legend(
                ["$σ^0_{}$(SAR)".format(self.pol), "$σ^0_{}$".format(self.pol) + "({})".format(self.model)])
            plt.savefig('model_time_series_{}_'.format(self.model) + '_{}_'.format(self.pol) + 'khrms_' +
                        '{:.2f}'.format((2 * np.pi / self.wavelength) * self.hrms_list[i]) + '.png')
            plt.show()
        # time series plot creation
