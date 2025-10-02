import glob
import pandas as pd


def del_red_tseries_rows(folder_ending, tseries_path, csv_ismn_files_path):
    tseries_list = []
    for nd in folder_ending:
        tseries_list.append(pd.read_csv(glob.glob(tseries_path.format(nd))[0]))

    ismn_files_fnames_list = []
    ismn_files_dates_list = []
    for nd, i in zip(folder_ending, range(len(folder_ending))):
        ismn_files_fnames_list.append(glob.glob(csv_ismn_files_path.format(nd))[0])
        ismn_files_dates_list.append((pd.read_csv(ismn_files_fnames_list[i], sep=' ', usecols=['Date'],
                                                  skip_blank_lines=True).squeeze("columns")))
        ismn_files_dates_list[i] = (ismn_files_dates_list[i].dropna(how='all')).to_list()

    for i in range(len(ismn_files_dates_list)):
        for j in range(len(ismn_files_dates_list[i])):
            ismn_files_dates_list[i][j] = ismn_files_dates_list[i][j][:4] + '-' + \
                                          ismn_files_dates_list[i][j][5:7] + '-' + \
                                          ismn_files_dates_list[i][j][8:]

    for i in range(len(tseries_list)):
        tseries_list[i] = tseries_list[i][tseries_list[i]['C0/date'].isin(ismn_files_dates_list[i])]

    return tseries_list

