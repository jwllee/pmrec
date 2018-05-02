#!/usr/bin/env python

import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix
import os, sys
import functools as fct
import pmRecUtils as rutils
from dataRep import nextActivityAsStep
from dataRep import nextActivityAsStepBuildWithVariant
from dataRep import nextActivityAsStepAddActivity
from dataRep import nextActivityAsStepAddActivityBuildWithVariant
from dataRep import nextActivity
from dataRep import nextActivityBuildWithVariant
from dataRep import nextActivityAddActivityBuildWithVariant
from dataRep import rStep as rstep
from dataRep import rStepLifecycle as rsteplc
from dataRep import rStepAmt as rstepamt
from dataRep import rAct as ract
from dataRep import rStepLifecyclePlus as rsteplcplus
from dataRep import rStepImpact as rstepimpact


'''
Module to convert event logs and other data format to required input format for fastFM
'''

# ARTIFICIAL START EVENT to know when a case has initiated
ARTIFICIAL_START = 'ARTIFICIAL_START'


def step_log_to_fm_format_all(func, *args, **kwargs):
    func_mapping = {'normal':nextActivityAsStep.step_log_to_fm_format, \
                    'activity':nextActivityAsStepAddActivity.step_log_to_fm_format, \
                    'normal_var':nextActivityAsStepBuildWithVariant.step_log_to_fm_format, \
                    'activity_var':nextActivityAsStepAddActivityBuildWithVariant.step_log_to_fm_format,\
                    'normal_1':nextActivity.step_log_to_fm_format, \
                    'normal_1_var':nextActivityBuildWithVariant.step_log_to_fm_format,\
                    'activity_1_var':nextActivityAddActivityBuildWithVariant.step_log_to_fm_format,\
                    'step':rstep.log_to_fm_format, \
                    'step_lc':rsteplc.log_to_fm_format, \
                    'step_amt':rstepamt.log_to_fm_format, \
                    'act':ract.log_to_fm_format, \
                    'step_lc_plus':rsteplcplus.log_to_fm_format, \
                    'step_impact':rstepimpact.log_to_fm_format}
    return func_mapping[func](*args, **kwargs)

if __name__ == '__main__':
    # load the small bpic2012 dataset
    log_small = './dataset/bpic2012/bpic2012FirstHundred.csv'
    log_small_df = pd.read_csv(log_small, index_col=False)
    print('Number of cases: {}, number of events: {}'\
          .format(log_small_df['caseId'].unique().shape[0], \
                     log_small_df['activity'].shape[0]))
    # caseid list and activity list
    caseid_list = log_small_df['caseId'].unique().tolist()
    activity_list = log_small_df['activity'].unique().tolist()
    activity_list = np.append([ARTIFICIAL_START,], activity_list)
    activity_list = sorted(activity_list)
    step_list = rutils.mk_possible_steps(activity_list, stepsz=2)
    print('num_of_acts: {} \n{}'.format(len(activity_list), activity_list))
    print('num_of_steps: {} \n{}'.format(len(step_list), step_list))

    print('type step: {} in steplist of type: {}'.format(type(step_list[0]), \
                                                         type(step_list)))
    # create mapping between stepid and step
    step_mapping = {step_list[i]:i for i in range(len(step_list))}
    rev_step_mapping = {val:key for key, val in step_mapping.items()} 

    # map everything
    step_list = np.asarray(list(map(lambda step: step_mapping[tuple(step)],\
                                    step_list)))

    # create next step mapping for each step
    get_next_step = \
            lambda step: list(filter(lambda step1: \
                                     rev_step_mapping[step1][0] == \
                                     rev_step_mapping[step][-1], step_list))
    next_step_mapping = {step: get_next_step(step) for step in step_list}

    # get a case and make it into step case representation
    caseid1 = caseid_list[0]
    case1 = log_small_df[(log_small_df['caseId']==caseid1)]['activity'].tolist()
    case1 = np.append([ARTIFICIAL_START,], case1)
    print('case {} has {} events: {}'.format(caseid1, len(case1), case1))
    step_case1 = np.asarray([(case1[i-1], case1[i]) \
                                for i in range(1, len(case1))])
    print('step case {} has {} steps: {}'.format(caseid1, len(step_case1),  step_case1))
    step_case1 = list(map(lambda step: step_mapping[tuple(step)], step_case1))
    step_case1 = np.asarray(step_case1)

    assert np.asarray(list(map(lambda step: step in step_list, \
                                  step_case1))).all()

    # create a matrix representation of step case
    x_datalist, x_row_inds, x_col_inds, x_shape, \
            y_datalist, pred_id_list = \
                        step_case_to_fm_format(caseid1, step_case1, caseid_list, \
                                   step_list, next_step_mapping, \
                                   minpartialsz=2, negative_samples=-1, \
                                   seed=123, normalize=True)

    print('Created matrix representation of step case {}'.format(caseid1))
    step_case1_mat = csc_matrix((x_datalist, (x_row_inds, x_col_inds)), x_shape)
    print('Dimension of step case: {}'.format(x_shape))
    print(step_case1_mat)

    # testing using the whole log
    # convert log to step log dict
    step_log = dict()
    for cid in caseid_list:
        case = log_small_df[(log_small_df['caseId']==cid)]['activity'].tolist()
        case = np.append([ARTIFICIAL_START,], case)
        print('case {} has {} events.'.format(cid, len(case)))
        step_case = np.asarray([(case[i-1], case[i]) \
                                for i in range(1, len(case))])
        step_case = list(map(lambda step: step_mapping[tuple(step)],\
                             step_case))
        step_case = np.asarray(step_case)

        step_log[cid] = step_case

    # convert to matrix representation of step log
    x_datalist_log, x_row_inds_log, x_col_inds_log, x_shape_log, \
            y_datalist_log, pred_id_list  = \
                step_log_to_fm_format(step_log, caseid_list, \
                                                   step_list, \
                                                   next_step_mapping, \
                                                   minpartialsz=2, \
                                                   negative_samples=3, \
                                                   seed=123, normalize=True)

    print('Created matrix representation of step log!')
    print('Dimension of step log: {}'.format(x_shape_log))
    step_log_mat = csc_matrix((x_datalist_log, (x_row_inds_log, \
                                                x_col_inds_log)), x_shape_log)
    print(step_log_mat)
    chosen_case = caseid_list[3]
    print('case {}: {}'.format(chosen_case, step_log_mat.todense()[3,100:625]))
    print('case {} has {} events'.format(chosen_case, \
                                         log_small_df[(log_small_df['caseId']==chosen_case)].shape[0]))


    # create a matrix representation of step log using activity style
    x_datalist_log1, x_row_inds_log1, x_col_inds_log1, x_shape_log1, \
            y_datalist_log1 = step_log_to_fm_format_1(step_log, caseid_list, \
                                  step_list, next_step_mapping, \
                                  activity_list, rev_step_mapping, \
                                  minpartialsz=2, \
                                  negative_samples=3, seed=123, \
                                  normalize=True)
    print('Created matrix representation of step log using activity mode!')
    print('Dimension of step log: {}'.format(x_shape_log1))

