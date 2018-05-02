import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix
import os, sys
import functools as fct

# import from parent directory
nb_dir = os.path.split(os.getcwd())[0]
if nb_dir not in sys.path:
    print('Appending directory as path: {}'.format(nb_dir))
    sys.path.append(nb_dir)

import pmRecUtils as rutils

'''
Builds log matrix from step log using features:
    - executed steps
and predicts the next step.
'''



def variant_to_fm_format(steps, step_list, next_step_mapping, minpartialsz=2,\
                           negative_samples=3, seed=123, \
                           normalize=True, pred_id=0):
    # create objects to be returned
    x_datalist = list()
    x_row_inds = list()
    x_col_inds = list()
    x_shape = np.zeros(shape=(2,))

    y_datalist = list()

    # list of pred_id to make later identification faster
    pred_id_list = list()

    # base step, steps is shorter or equal to minimum partial size, 
    # return it since no prediction to be made
    if steps.shape[0] <= minpartialsz:
        return np.asarray(x_datalist), np.asarray(x_row_inds), \
                np.asarray(x_col_inds), x_shape, \
                np.asarray(y_datalist), np.asarray(pred_id_list)

    for ind in range(minpartialsz, steps.shape[0]):
        partialcase = steps[:ind]
        laststep = steps[ind - 1]
        possible_next_step_list = next_step_mapping[laststep]

        assert len(possible_next_step_list) > 0

        # actual next step
        gt_next_step = steps[ind]

        # sample negative samples
        if negative_samples > -1:
            np.random.seed(seed=seed)
            random_negative_samples = \
                    list(filter(lambda step: step != gt_next_step, \
                                                  possible_next_step_list))
            picked_inds = np.random.choice(np.arange(len(random_negative_samples)), \
                                                       size=negative_samples, \
                                                       replace=False)
            random_negative_samples = list(map(lambda ind: \
                                               random_negative_samples[ind], \
                                               picked_inds))
            samples = np.append(random_negative_samples, [gt_next_step,])
        else:
            # select all the possible next steps
            samples = list(filter(lambda step: step != gt_next_step, \
                                  possible_next_step_list))
            samples = np.append(samples, [gt_next_step,])

        # create the taken steps part
        taken_repeat = np.asarray([partialcase for _ in range(len(samples))])
        taken_datalist, taken_row_inds, taken_col_inds, taken_shape = \
                rutils.multiple_to_fm_format(taken_repeat, step_list, normalize)

        # samples
        next_datalist, next_row_inds, next_col_inds, next_shape = \
                rutils.single_to_fm_format(samples, step_list)
   
        # create the matrix representation of step case in fm format
        # by combining all the above info
        # check dimensions
        assert taken_shape[1] == len(step_list), \
                'taken shape: {}, step list shape: {}'\
                .format(taken_shape, len(step_list))
        assert next_shape[1] == len(step_list), \
                'next shape: {}, step list shape: {}'\
                .format(next_shape, len(step_list))

        # shift next columns by |step_list|
        next_col_inds = next_col_inds + len(step_list)

        x_datalist_i = np.concatenate((taken_datalist, next_datalist))
        x_row_inds_i = np.concatenate((taken_row_inds, next_row_inds))
        x_col_inds_i = np.concatenate((taken_col_inds, next_col_inds))
        x_shape_i = np.asarray((len(samples), 2 * len(step_list)))

    #    print('creating x shape: {}'.format(x_shape))

        # create the target y datalist, putting 1 for the next step
        # and putting 0 for the negative samples
        y_datalist_i = np.asarray([np.int(step) for step in samples == gt_next_step])
        
    #    print('y_datalist: {}'.format(y_datalist))

        # pred_id_list, should all be the same prediction number 
        pred_id_list_i = np.ones(len(y_datalist_i)) * pred_id

        # append to current results and shift results if needed
        if len(x_datalist) > 0:
            assert x_shape[1] == x_shape_i[1], \
                    'x_shape: {} not equal x_shape_i: {}'\
                    .format(x_shape, x_shape_i)

            # shift rows of x_row_inds_i by number of existing rows
            x_row_inds_i = x_row_inds_i + x_shape[0]
            
            x_datalist = np.concatenate((x_datalist, x_datalist_i))
            x_row_inds = np.concatenate((x_row_inds, x_row_inds_i))
            x_col_inds = np.concatenate((x_col_inds, x_col_inds_i))
            x_shape = np.asarray((x_shape[0] + x_shape_i[0], x_shape[1]))

            y_datalist = np.concatenate((y_datalist, y_datalist_i))

            pred_id_list = np.concatenate((pred_id_list, pred_id_list_i))
        else:
            x_datalist = x_datalist_i
            x_row_inds = x_row_inds_i
            x_col_inds = x_col_inds_i
            x_shape = x_shape_i
            y_datalist = y_datalist_i
            pred_id_list = pred_id_list_i

        # increment pred id
        pred_id += 1

    return x_datalist, x_row_inds, x_col_inds, x_shape, y_datalist, \
        pred_id_list


def step_log_to_fm_format(var_to_cid, var_to_step, caseid_list, \
                            step_list, next_step_mapping, \
                            minpartialsz=2, negative_samples=3, \
                            seed=123, normalize=True, pred_id=0):
    # create the result objects
    x_datalist = list()
    x_row_inds = list()
    x_col_inds = list()
    x_shape = np.zeros(2)
    y_datalist = list()

    # prediction id list
    pred_id_list = list()

    for vid, steps in var_to_step.items():
        x_datalist_v, x_row_inds_v, x_col_inds_v, \
                x_shape_v, y_datalist_v, pred_id_list_v = \
                variant_to_fm_format(steps, step_list, \
                                     next_step_mapping, \
                                     minpartialsz=minpartialsz, \
                                     negative_samples=negative_samples, \
                                     seed=seed, normalize=normalize, \
                                     pred_id=pred_id)
        # repeat it for each case_id
        var_cid_list = var_to_cid[vid]
        for i in range(len(var_cid_list)):
            cid = var_cid_list[i]
            # create a list of zeros with 1 at the position
            # of cid
            cid_pos = np.where(np.asarray(caseid_list)==cid)[0]
            assert cid_pos.size == 1
            cid_pos = cid_pos[0]
            # create datalist, row_inds and etc
            # repeat x_shape_v[0] times, for each row
            to_repeat = x_shape_v[0]
            x_datalist_cid = np.ones(to_repeat)
            x_row_inds_cid = np.arange(to_repeat)
            x_col_inds_cid = np.ones(to_repeat) * cid_pos
            x_shape_cid = np.asarray((to_repeat, len(caseid_list)))

            # update the variant datalist, row_inds, and etc
            # put them at the start of the matrix
            # shift the columns of variant by |caseid_list|
            col_aux = x_col_inds_v + len(caseid_list)
            # shift the rows of case variant by i * to_repeat
            row_aux = x_row_inds_v + (to_repeat * i)

            x_datalist_cid = np.concatenate((x_datalist_cid, x_datalist_v))
            x_row_inds_cid = np.concatenate((x_row_inds_cid, row_aux))
            x_col_inds_cid = np.concatenate((x_col_inds_cid, col_aux))
            x_shape_cid = np.asarray((x_shape_v[0], x_shape_v[1] + \
                                    len(caseid_list)))
            y_datalist_cid = y_datalist_v.copy()
            # shift pred_id_list
            num_of_preds_v = len(set(pred_id_list_v))
            pred_id_list_cid = pred_id_list_v + num_of_preds_v * i

            # update log datalist, row_inds, and etc
            if len(x_datalist) == 0:
                # just replace
                x_datalist = x_datalist_cid
                x_row_inds = x_row_inds_cid
                x_col_inds = x_col_inds_cid
                x_shape = x_shape_cid
                y_datalist = y_datalist_cid
                pred_id_list = pred_id_list_cid
            else:
                # assert that the number of columns are the same
                assert x_shape[1] == x_shape_cid[1], \
                        'x_shape: {}, x_shape_cid: {}'\
                        .format(x_shape, x_shape_cid)
                # shift row_inds
                x_row_inds_cid = x_row_inds_cid + x_shape[0]
                # update results
                x_datalist = np.concatenate((x_datalist, x_datalist_cid))
                x_row_inds = np.concatenate((x_row_inds, x_row_inds_cid))
                x_col_inds = np.concatenate((x_col_inds, x_col_inds_cid))
                x_shape = np.asarray((x_shape[0] + x_shape_cid[0], x_shape[1]))
                y_datalist = np.concatenate((y_datalist, y_datalist_cid))
                # update pred_id_list
                pred_id_list = np.concatenate((pred_id_list, pred_id_list_cid))
        # update pred_id
        pred_id = pred_id_list[-1] + 1
    return x_datalist, x_row_inds, x_col_inds, x_shape, y_datalist, \
            pred_id_list


