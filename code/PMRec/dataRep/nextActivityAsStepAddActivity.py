#!/usr/bin/env python

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


def step_case_to_fm_format(caseid, steps, caseid_list, step_list,
                           next_step_mapping, activity_list, \
                           rev_step_mapping, minpartialsz=2,\
                           negative_samples=3, seed=123, \
                           normalize=True, pred_id=0):
    '''
    Method to make matrix representation of step case. Matrix will contain
    the following features:
        - caseIds
        - steps taken
        - executed activities
        - last executed activity
        - step to be predicted
    '''
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
                
        # repeat caseids for |samples| times
        cid_repeat = np.asarray([caseid,]).repeat(len(samples))
        cid_datalist, cid_row_inds, cid_col_inds, cid_shape = \
                rutils.single_to_fm_format(cid_repeat, caseid_list)
        
        # create the taken steps part
        taken_repeat = np.asarray([partialcase for _ in range(len(samples))])
        taken_datalist, taken_row_inds, taken_col_inds, taken_shape = \
                rutils.multiple_to_fm_format(taken_repeat, step_list, normalize)

        # create the taken activities part
        t_acts = list(map(lambda step: rev_step_mapping[step], partialcase))
        t_acts = list(map(lambda step: step[-1], t_acts))
        t_acts = np.asarray(t_acts)
        # need to add the first activity of the first step
        t_acts = np.append([ARTIFICIAL_START,], t_acts)
    #    print('t_acts: {}'.format(t_acts))
        not_in = list(filter(lambda act: act not in activity_list, t_acts))
        assert len(not_in) ==  0, 't_acts not in activity list: \
            {} with {} items.'.format(str(not_in), len(not_in))
        t_acts_repeat = np.asarray([t_acts for _ in range(len(samples))])
        t_acts_datalist, t_acts_row_inds, t_acts_col_inds, t_acts_shape = \
                rutils.multiple_to_fm_format(t_acts_repeat, activity_list, normalize)

        # create the last executed activity part
        l_act = rev_step_mapping[laststep][-1]
        assert l_act in activity_list, '{} not in activity list: {}'\
                .format(l_act, activity_list)
        l_act_repeat = np.asarray([l_act,]).repeat(len(samples))
        l_act_datalist, l_act_row_inds, l_act_col_inds, l_act_shape = \
                rutils.single_to_fm_format(l_act_repeat, activity_list)

        # samples
        next_datalist, next_row_inds, next_col_inds, next_shape = \
                rutils.single_to_fm_format(samples, step_list)
       
        # create the matrix representation of step case in fm format
        # by combining all the above info
        # check dimensions
        assert (cid_shape[0] == taken_shape[0]) and \
                (taken_shape[0] == next_shape[0]) and \
                (next_shape[0] == t_acts_shape[0]) and \
                (t_acts_shape[0] == l_act_shape[0]), \
                'cid shape: {}, taken shape: {}, \
                next shape: {}, t_acts shape: {}, \
                l_act shape: {}'.format(cid_shape, taken_shape, \
                                        next_shape, t_acts_shape, \
                                       l_act_shape)
        assert cid_shape[1] == len(caseid_list), \
                'cid shape: {}, taken shape: {}, next shape: {}'\
                .format(cid_shape, taken_shape, next_shape)
        assert taken_shape[1] == len(step_list), \
                'taken shape: {}, step list shape: {}'\
                .format(taken_shape, len(step_list))
        assert next_shape[1] == len(step_list), \
                'next shape: {}, step list shape: {}'\
                .format(next_shape, len(step_list))
        assert t_acts_shape[1] == len(activity_list), \
                'taken acts shape: {}, activity list shape: {}'\
                .format(t_acts_shape, len(activity_list))
        assert l_act_shape[1] == len(activity_list), \
                'last act shape: {}, activity list shape: {}'\
                .format(l_act_shape, len(activity_list))

        # shift taken columns by |caseid_list|
        taken_col_inds = taken_col_inds + len(caseid_list)
        # shift taken activities columns by |caseid_list| + |step_list|
        t_acts_col_inds = t_acts_col_inds + len(caseid_list) + len(step_list)
        # shift last taken activity columns by |caseid_list| + |step_list| + 
        # |activity_list|
        l_act_col_inds = l_act_col_inds + len(caseid_list) + len(step_list) + \
                len(activity_list)
        # shift next columns by |caseid_list| + |step_list| + 2 * |activity_list|
        next_col_inds = next_col_inds + len(caseid_list) + len(step_list) + \
                (len(activity_list) * 2)

        x_datalist_i = np.concatenate((cid_datalist, taken_datalist, \
                                     t_acts_datalist, l_act_col_inds, \
                                     next_datalist))
        x_row_inds_i = np.concatenate((cid_row_inds, taken_row_inds, \
                                     t_acts_row_inds, l_act_row_inds, \
                                     next_row_inds)) 
        x_col_inds_i = np.concatenate((cid_col_inds, taken_col_inds, \
                                     t_acts_col_inds, l_act_col_inds, \
                                     next_col_inds))
        num_of_cols = len(caseid_list) + 2 * len(step_list) + 2 * \
            activity_list.shape[0]  
        x_shape_i = np.asarray((len(samples), num_of_cols))

    #    print('creating x shape: {}'.format(x_shape))

        # create the target y datalist, putting 1 for the next step
        # and putting 0 for the negative samples
        y_datalist_i = np.asarray([np.int(step) for step in samples == gt_next_step])
        
        print('y_datalist: {}'.format(y_datalist))
        assert y_datalist_i[-1] == 1, \
                'next step: {}, \nsamples: {}'\
                .format(gt_next_step, samples)

        # pred_id_list, should all be the same prediction number 
        pred_id_list_i = np.ones(len(y_datalist_i)) * pred_id

        # check the dimensions make sense
        # column should be the same
        if len(x_datalist) > 0:
            assert x_shape[1] == x_shape_i[1], \
                    'x_shape: {} not equal x_shape_i: {}'\
                    .format(x_shape, x_shape_i)
            # shift rows of x_row_inds1 by number of existing rows
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

        pred_id += 1

    return x_datalist, x_row_inds, x_col_inds, x_shape, y_datalist, \
        pred_id_list


def step_log_to_fm_format(step_log, caseid_list, step_list, \
                         next_step_mapping, activity_list, \
                           rev_step_mapping,  minpartialsz=2, \
                         negative_samples=3, seed=123, \
                         normalize=True, pred_id=0):
    # create the result objects
    x_datalist = list()
    x_row_inds = list()
    x_col_inds = list()
    x_shape = np.zeros(2)
    y_datalist = list()

    # prediction id list
    pred_id_list = list()

    for cid in step_log.keys():
        step_case = step_log[cid]
        x_datalist_cid, x_row_inds_cid, x_col_inds_cid, \
                x_shape_cid, y_datalist_cid, pred_id_list_cid = \
                step_case_to_fm_format(cid, step_case, caseid_list, \
                                       step_list, next_step_mapping, \
                                      activity_list, rev_step_mapping, \
                                       minpartialsz=minpartialsz, \
                                       negative_samples=negative_samples, \
                                       seed=seed, normalize=normalize, \
                                      pred_id=pred_id)
        if len(x_datalist) > 0:
            # check dimensions are correct
            assert x_shape[1] == x_shape_cid[1], \
                    'x_shape: {}, x_shape_cid: {}'.format(x_shape,\
                                                          x_shape_cid)

        # need to shift rows downwards by current number of rows
        x_row_inds_cid = x_row_inds_cid + x_shape[0]

        # update result
        x_datalist = np.concatenate((x_datalist, x_datalist_cid))
        x_row_inds = np.concatenate((x_row_inds, x_row_inds_cid))
        x_col_inds = np.concatenate((x_col_inds, x_col_inds_cid))
        x_shape = np.asarray([x_shape[0] + x_shape_cid[0], x_shape_cid[1]])
        y_datalist = np.concatenate((y_datalist, y_datalist_cid))

        pred_id_list = np.concatenate((pred_id_list, pred_id_list_cid))

        # update prediction id
        pred_id = pred_id_list[-1] + 1

    return x_datalist, x_row_inds, x_col_inds, x_shape, y_datalist, \
            pred_id_list


