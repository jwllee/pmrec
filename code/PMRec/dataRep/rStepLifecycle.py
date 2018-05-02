#!/usr/bin/env python
import pandas as pd
import numpy as np
from scipy.sparse import csc_matrix
import os, sys
import functools as fct

# import from parent directory
nb_dir = os.path.split(os.getcwd())[0]
if nb_dir not in sys.path:
    print('Appending directory as path: {}'.format(nb_dir))
    sys.path.append(nb_dir)

import pmRecUtils as rutils
import logUtils as lutils


def partialcase_to_fm_format(caseid, partialcase, partialcase_lifecycle, \
                             stepsz, cid_list, \
                             step_dict, step_dict_lifecycle, \
                             next_step_mapping, \
                             negative_samples, seed, \
                             normalize, pred_id):
    x_datalist = list()
    x_row_inds = list()
    x_col_inds = list()
    x_shape = np.zeros(shape=(2,))

    y_datalist = list()

    pred_id_list = list()

    if partialcase.shape[0] <= 2:
        # not enough events to make a step
        return np.asarray(x_datalist), np.asarray(x_row_inds), \
                np.asarray(x_col_inds), np.asarray(x_shape), \
                np.asarray(y_datalist), np.asarray(pred_id_list)
    # there should negative samples + 1 rows
    # negative samples might be too big
    if negative_samples >= 0 and \
       negative_samples < len(list(next_step_mapping.keys())):
        num_of_rows = negative_samples + 1
    else:
        num_of_rows = len(list(next_step_mapping.keys()))
    x_shape[0] = num_of_rows
    all_steps = fct.reduce(lambda x, y: x + y.shape[0], step_dict_lifecycle.values(), 0)
    x_shape[1] = cid_list.shape[0] + all_steps + step_dict[2].shape[0]
    x_shape = x_shape.astype(np.int)

    # make into 2-steps
    steps = lutils.case_to_steps(partialcase, 2)
    partialsteps = steps[:-1]
    to_predict = steps[-1]
    possible = next_step_mapping[partialcase[-2]]

    # pick negative samples
    if negative_samples > -1:
        rand_negatives = filter(lambda s: s != to_predict, possible)
        # need to check if negative_samples is larger than len(rand_negatives)
        rand_negatives = list(rand_negatives)
        random_sz = negative_samples
        if negative_samples > len(rand_negatives):
            # use the size of rand_negatives if this is bigger
            random_sz = len(rand_negatives)
        rand_negatives = np.random.choice(list(rand_negatives), \
                                       size=random_sz, \
                                       replace=False)
        samples = np.append(rand_negatives, [to_predict,])
    else:
        samples = filter(lambda s: s != to_predict, possible)
        samples = np.append(list(samples), [to_predict,])

    # create block for cid
    cid_repeat = np.asarray([caseid,]).repeat(len(samples))
    cid_datalist, cid_row_inds, cid_col_inds, cid_shape = \
            rutils.single_to_fm_format(cid_repeat, cid_list)

    # create block for taken steps
    taken_datalist, taken_row_inds, taken_col_inds, taken_shape = \
            rutils.mk_step_block(partialcase_lifecycle[:-1], \
                                 step_dict_lifecycle, stepsz, \
                                 repeat=len(samples), \
                                 normalize=normalize)

    # create block for samples
#    print('Samples: {}'.format(samples))
    next_datalist, next_row_inds, next_col_inds, next_shape = \
            rutils.single_to_fm_format(samples, step_dict[2])

    # shift taken columns by |cid_list|
    taken_col_inds = taken_col_inds + cid_list.shape[0]

    # shift next columns by |cid_list| + |step_list|
    next_col_inds = next_col_inds + cid_list.shape[0] + all_steps

    x_datalist = np.concatenate((cid_datalist, taken_datalist, next_datalist))
    x_row_inds = np.concatenate((cid_row_inds, taken_row_inds, next_row_inds))
    x_col_inds = np.concatenate((cid_col_inds, taken_col_inds, next_col_inds))
    x_row_inds = x_row_inds.astype(np.int)
    x_col_inds = x_col_inds.astype(np.int)

    y_datalist = np.asarray([np.int(step) for step in samples == to_predict], \
                            dtype=np.int)

    pred_id_list = np.ones(len(y_datalist)) * pred_id

    return x_datalist, x_row_inds, x_col_inds, x_shape, \
            y_datalist, pred_id_list


def case_to_fm_format(caseid, case, case_lc, stepsz, cid_list, \
                      step_dict, step_dict_lc, \
                     next_step_mapping, minpartialsz, \
                     negative_samples, seed, \
                     normalize, pred_id):
    # create objects to be returned
    x_datalist = list()
    x_row_inds = list()
    x_col_inds = list()
    x_shape = np.zeros(shape=(2,), dtype=np.int)

    y_datalist = list()

    pred_id_list = list()

    # need to have minpartialsz plus one to predict
    if case.shape[0] <= minpartialsz:
        return np.asarray(x_datalist), np.asarray(x_row_inds), \
                np.asarray(x_col_inds), x_shape, \
                np.asarray(y_datalist), np.asarray(pred_id_list)

    for ind in range(minpartialsz, case.shape[0] + 1):
        partialcase = case[:ind]
        partialcase_lc = case_lc[:ind]
#        print('building for partial case: {}'.format(partialcase))
        x_datalist_i, x_row_inds_i, x_col_inds_i, x_shape_i, \
                y_datalist_i, pred_id_list_i = \
                partialcase_to_fm_format(caseid, partialcase, \
                                         partialcase_lc, \
                                         stepsz, cid_list, \
                                         step_dict, step_dict_lc, \
                                         next_step_mapping, \
                                         negative_samples, seed, \
                                         normalize, pred_id)
        # shift by rows if necessary
        if len(x_datalist) == 0:
            x_datalist = x_datalist_i
            x_row_inds = x_row_inds_i
            x_col_inds = x_col_inds_i
            x_shape = x_shape_i
            y_datalist = y_datalist_i
            pred_id_list = pred_id_list_i
        else:
            x_row_inds_i += x_shape[0]

            x_datalist = np.concatenate((x_datalist, x_datalist_i))
            x_row_inds = np.concatenate((x_row_inds, x_row_inds_i))
            x_col_inds = np.concatenate((x_col_inds, x_col_inds_i))
            x_shape = np.asarray((x_shape[0] + x_shape_i[0], x_shape[1]))
            y_datalist = np.concatenate((y_datalist, y_datalist_i))
            pred_id_list = np.concatenate((pred_id_list, pred_id_list_i))

        # update pred_id
        if len(pred_id_list) > 0:
            pred_id = pred_id_list[-1] + 1

    return x_datalist, x_row_inds, x_col_inds, x_shape, \
            y_datalist, pred_id_list



def mk_case_lc(case_with_lc):
    case_lc = list()
    for event in case_with_lc:
        filtered = filter(lambda e: str(e)!='nan', event)
        joined = '+'.join(filtered)
        case_lc.append(joined)
    return np.asarray(case_lc)


def log_to_fm_format(log, stepsz, cid_list, step_dict, \
                     step_dict_lc, next_step_mapping, \
                     minpartialsz=3, negative_samples=3, seed=123, \
                     normalize=True, pred_id=0):
    # create objects to be returned
    x_datalist = list()
    x_row_inds = list()
    x_col_inds = list()
    x_shape = np.zeros(2, dtype=np.int)

    y_datalist = list()

    pred_id_list = list()

    log_caseids = log['caseId'].unique()
    log_caseids.sort()
    np.random.seed(seed=seed)
    for cid in log_caseids:
#        print('Building for case: {}'.format(cid))
        case_df = log[(log['caseId']==cid)]
        case = case_df['activity'].values
        case_lc = mk_case_lc(case_df[['activity', 'lifecycle']].values)

        x_datalist_i, x_row_inds_i, x_col_inds_i, x_shape_i, \
                y_datalist_i, pred_id_list_i = \
                case_to_fm_format(cid, case, case_lc, stepsz, cid_list, \
                                  step_dict, step_dict_lc, next_step_mapping, \
                                  minpartialsz, negative_samples, \
                                  seed, normalize, pred_id)
        if len(x_datalist) == 0:
            x_datalist = x_datalist_i
            x_row_inds = x_row_inds_i
            x_col_inds = x_col_inds_i
            x_shape = x_shape_i
            y_datalist = y_datalist_i
            pred_id_list = pred_id_list_i
        else:
            # shift by rows
            x_row_inds_i += x_shape[0]

            x_datalist = np.concatenate((x_datalist, x_datalist_i))
            x_row_inds = np.concatenate((x_row_inds, x_row_inds_i))
            x_col_inds = np.concatenate((x_col_inds, x_col_inds_i))
            x_shape = np.asarray((x_shape[0] + x_shape_i[0], x_shape[1]))
            y_datalist = np.concatenate((y_datalist, y_datalist_i))
            pred_id_list = np.concatenate((pred_id_list, pred_id_list_i))

        # update pred_id
        if len(pred_id_list) > 0:
            pred_id = pred_id_list[-1] + 1

    return x_datalist, x_row_inds, x_col_inds, x_shape, \
            y_datalist, pred_id_list

