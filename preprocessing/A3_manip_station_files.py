import glob
import os


class manip_ismn_files:
    """
    Class that contains functions for reformating the downloaded ISMN Data Files so that they can be converted to
    .csv format
    """

    def __init__(self, folds_path, depth_instr_meas, compl_txt_path, compl_csv_path, no_mult_meas_depths):
        """

        Args:
            folds_path: a string which contains the path to the folders where ISMN data is stored (ex. C:/FR_Aqui/*)
            downloaded in the Header+values format

            depth_instr_meas: a tuple which contains lists with the soil moisture and temperature measurement depths for
            each ISMN station as well as measurement instruments that are not needed (for ex. for 2 stations with 2
            measurement depths not needed we might get the following tuple: (['0.10', '0.20', '0.05*Thermistor-linear'],
            ['0.09', '0.21])

            compl_txt_path: a string which contains the full path to the .txt files (for ex. C:/FR_Aqui/*_{}/*.txt)

            compl_csv_path: a string which contains the full path to the .csv files (for ex. C:/FR_Aqui/*_{}/*.csv)

            no_mult_meas_depths: a boolean which is set to True if the soil moisture and temperature was measured in
            one only depth

        """
        self.folds_path = folds_path
        self.depth_instr_meas = depth_instr_meas
        self.compl_txt_path = compl_txt_path
        self.compl_csv_path = compl_csv_path
        self.no_mult_meas_depths = no_mult_meas_depths

    def prepare_raw_ismn_data(self):
        """

        Returns: does not return an object but changes the file extension of the ISMN data and filters them so that they
        do not contain data from measurement depths that are not appropriate for the application for which they are used

        """
        ismn_folds = glob.glob(self.folds_path)
        ismn_stm_files = []
        # create a list with the paths to the station folders

        for fold, n in zip(ismn_folds, range(len(ismn_folds))):
            os.rename(fold, fold + '_{}'.format(n))
            ismn_stm_files.append(glob.glob(self.folds_path + '_{}/*.stm'.format(n)))
            # rename the station folder names so that they end with an integer in the range of
            # (0, N_stations)

        for n in range(len(ismn_stm_files)):
            for f in range(len(ismn_stm_files[n])):
                os.rename(ismn_stm_files[n][f], ismn_stm_files[n][f][:-3] + 'txt')
                # change the file extension of the ISMN data files from .stm to .txt

        if not self.no_mult_meas_depths:
            ismn_folds_stm_remove_dpth_instr = []
            # initialization of a list where the soil moisture and temperature files that will be removed are located

            for n in range(len(self.depth_instr_meas)):
                for m in range(len(self.depth_instr_meas[n])):
                    ismn_folds_stm_remove_dpth_instr.append(glob.glob(self.folds_path + '_{}/*{}*'.
                                                                      format(n, self.depth_instr_meas[n][m])))

            for i in range(len(ismn_folds_stm_remove_dpth_instr)):
                for j in range(len(ismn_folds_stm_remove_dpth_instr[i])):
                    if ismn_folds_stm_remove_dpth_instr[i][j] != []:
                        os.remove(ismn_folds_stm_remove_dpth_instr[i][j])
        else:
            pass

    def manip_ismn_files_lines(self):
        """

        Returns: does not return an object but formats the .txt files so that the data are sorted by columns and can be
        converted to .csv files

        """
        txt_ismn_files = []
        for n in range(len(glob.glob(self.folds_path))):
            txt_ismn_files.append(glob.glob(self.compl_txt_path.format(str(n))))

        for i in range(len(txt_ismn_files)):
            for ismn_file in txt_ismn_files[i]:
                with open(ismn_file, "r") as ismn_reader:
                    ismn_lines_list = []
                    for sm_line in ismn_reader.readlines():
                        ismn_lines_list.append(sm_line)
                        # for an ISMN station folder read the lines of the files and append them to a list #

                    if '_sm_' in ismn_file:
                        ismn_lines_list[0] = 'Date Time SM_(m^3/m^3) Quality_Flag_1 Quality_Flag_2'
                        ismn_lines_list.pop(1)
                    elif '_ts_' in ismn_file:
                        ismn_lines_list[0] = 'Date Time Soil_Temp_(Celsius) Quality_Flag_1 Quality_Flag_2'
                        ismn_lines_list.pop(1)
                    elif '_ta_' in ismn_file:
                        ismn_lines_list[0] = 'Date Time Air_Temp_(Celsius) Quality_Flag_1 Quality_Flag_2'
                        ismn_lines_list.pop(1)
                    elif '_p_' in ismn_file:
                        ismn_lines_list[0] = 'Date Time Precip_(mm) Quality_Flag_1 Quality_Flag_2'
                        ismn_lines_list.pop(1)
                    # delete the two first rows and replace the contents of the first row with a header (slightly diffe-
                    # rent for the soil moisture and temperature datafiles)

                    remove_line = []
                    remove_5spaces = []
                    remove_3spaces = []
                    remove_2spaces = []

                    for lin in ismn_lines_list:
                        rem_lin = lin.replace('\n', '')
                        remove_line.append(rem_lin)

                    for lin in remove_line:
                        repl_5_spaces = lin.replace('     ', ' ')
                        remove_5spaces.append(repl_5_spaces)

                    for lin in remove_5spaces:
                        repl_3_spaces = lin.replace('   ', ' ')
                        remove_3spaces.append(repl_3_spaces)

                    for lin in remove_3spaces:
                        repl_2_spaces = lin.replace('  ', ' ')
                        remove_2spaces.append(repl_2_spaces)
                        # modification of the lists that contain the lines for the SM and ST data files so that the #
                        # data is sorted into columns #

                with open(ismn_file, "w") as ismn_writer:
                    for f_lin in remove_2spaces:
                        ismn_writer.write(f_lin)
                        ismn_writer.write("\n")

    def ch_fext(self):
        """

        This functionality changes the extensions of the files from .txt to .csv and the extensions of the .csv files
        present in the folder (the static variables) to .xlsx so that they will not be confused with the data files

        """
        txt_files_list = []
        csv_files_list = []

        for n in range(len(glob.glob(self.folds_path))):
            txt_files_list.append(glob.glob(self.compl_txt_path.format(str(n))))
            csv_files_list.append(glob.glob(self.compl_csv_path.format(str(n))))
            # create a list with the paths to the .txt files

        for csv in csv_files_list:
            os.rename(csv[0], csv[0][:-4] + '.xlsx')
            # change the file extension of the static variables .csv file to .xlsx (please change the same file
            # extensions to .csv again after the A4_filter_sm_ts_values.py because the static variables files cannot be
            # read as .xlsx files)

        for i in range(len(txt_files_list)):
            for file in txt_files_list[i]:
                os.rename(file, file[:-4] + '.csv')
                # change the file extension of the .txt files to .csv
