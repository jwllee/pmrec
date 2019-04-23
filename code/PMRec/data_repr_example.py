import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix
import os, sys
import functools as fct
import pmRecUtils as rutils
from dataRep import rStep


ARTIFICIAL_START = '0'


if __name__ == '__main__':
    fp = '../../data/BPIC2012.csv'
    log_df = pd.read_csv(fp, index_col=False)

    log_df['a'] = log_df['activity']
    log_df['activity'] = log_df['activity_id'].astype(str)

    activity_list = log_df['activity'].unique().tolist()
    activity_list = np.append([ARTIFICIAL_START,], activity_list)
    step_list = rutils.mk_possible_steps(activity_list, stepsz=2)

    stepsz = 2
    # steps from activity list
    step_dict = dict()
    for sz in range(2, stepsz + 1):
        step_list_i = rutils.mk_possible_steps(activity_list, sz)
        step_dict[sz] = step_list_i

    # caseid list
    cid_list = log_df['caseId'].unique()
    cid_list.sort()
    next_step_mapping = dict()
    for a in activity_list:
        possible = filter(lambda s: s.split('->')[0] == a, step_dict[2])
        next_step_mapping[a] = np.asarray(list(possible))

    filtered_df = log_df.iloc[:100,:]
    formatted = rStep.log_to_fm_format(filtered_df, stepsz, cid_list,
                                       step_dict, next_step_mapping)

    x_datalist = formatted[0]
    x_row_inds = formatted[1]
    x_col_inds = formatted[2]
    x_shape = formatted[3]
    y_datalist = formatted[4]
    pred_id_list = formatted[5]
