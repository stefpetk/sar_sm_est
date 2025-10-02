import glob
import os
import pandas as pd
import numpy as np


class manip_grd_fs:
    """
    Class that contains functions that manipulate the S1 GRD images in their respective folders
    """

    def __init__(self, folder_ending, grd_image_path, grd_ims_folder_path, bckslash_n, csv_ismn_files_path):
        self.folder_ending = folder_ending
        self.grd_image_path = grd_image_path
        self.grd_ims_folder_path = grd_ims_folder_path
        self.bckslash_n = bckslash_n
        self.csv_ismn_files_path = csv_ismn_files_path

    def ch_grd_fnames(self):
        """
        Function that given a folder where multiple Sentinel 1 G.R.D. Images are stored sorts the images according to the
        acquisition date of the images

        Returns: does not return an object
        """
        for n in self.folder_ending:
            img_list = glob.glob(self.grd_image_path.format(n))
            fold_path = glob.glob(self.grd_ims_folder_path.format(n))[0]
            temp_img_list = [img_list[i].split('\\')[self.bckslash_n] for i in range(len(img_list))]
            final_img_list = [sl[17:25] + "_" + sl for sl in temp_img_list]

            for i in range(len(img_list)):
                os.rename(img_list[i], fold_path + '\\' + final_img_list[i])

    def del_red_grd_ims(self):
        """
        Function that given the acquisition dates deletes GRD images that are redundant
        Returns: does not return an object

        """

        ims_acq_dates_list = []
        for n, i in zip(self.folder_ending, range(len(self.folder_ending))):
            ims_acq_dates_list.append(glob.glob(self.grd_image_path.format(n)))
            for j in range(len(ims_acq_dates_list[i])):
                ims_acq_dates_list[i][j] = (ims_acq_dates_list[i][j].split('\\', self.bckslash_n)[self.bckslash_n])[:8]

        ismn_files_fnames_list = []
        ismn_files_dates_list = []
        for n, i in zip(self.folder_ending, range(len(self.folder_ending))):
            for j in range(len(glob.glob(self.csv_ismn_files_path.format(n)))):
                if 'sm' in glob.glob(self.csv_ismn_files_path.format(n))[j]:
                    ismn_files_fnames_list.append(glob.glob(self.csv_ismn_files_path.format(n))[j])
                    ismn_files_dates_list.append(pd.read_csv(ismn_files_fnames_list[i], sep=' ', usecols=['Date']).
                                                 squeeze("columns"))
                    ismn_files_dates_list[i] = (ismn_files_dates_list[i].dropna(how='all')).to_list()

        for i in range(len(ismn_files_dates_list)):
            for j in range(len(ismn_files_dates_list[i])):
                ismn_files_dates_list[i][j] = ismn_files_dates_list[i][j][:4] + \
                                              ismn_files_dates_list[i][j][5:7] + \
                                              ismn_files_dates_list[i][j][8:]

        red_grd_ims_list = []
        for i in range(len(ismn_files_dates_list)):
            red_grd_ims_list.append([])
            for j in range(len(ims_acq_dates_list[i])):
                if ims_acq_dates_list[i][j] not in ismn_files_dates_list[i]:
                    red_grd_ims_list[i].append(ims_acq_dates_list[i][j])
                else:
                    pass

        grd_ims_paths = []
        grd_ims_lists = []
        for n, i in zip(self.folder_ending, range(len(self.folder_ending))):
            grd_ims_paths.append(glob.glob(self.grd_image_path.format(n)))
            grd_ims_lists.append([])
            for j in range(len(grd_ims_paths[i])):
                grd_ims_lists[i].append(grd_ims_paths[i][j].split('\\', self.bckslash_n)[self.bckslash_n])

        for i in range(len(grd_ims_paths)):
            for j in range(len(grd_ims_paths[i])):
                if grd_ims_lists[i][j][:8] in red_grd_ims_list[i]:
                    os.remove(grd_ims_paths[i][j])


