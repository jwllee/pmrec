#!/usr/bin/env python

import numpy as np
import pandas as pd
import os, sys, time
import pmRecUtils as pmutils
import dataRepresentation as datrep
from collections import defaultdict as ddict
import itertools as itools


'''
This module is for useful log conversion methods
'''

def case_to_steps(case, sz):
    if case.shape[0] < sz:
        return np.asarray([])
    else:
        steps = list(map(lambda i: case[i:i+sz].copy(), \
                         np.arange(case.shape[0] - sz + 1)))
        steps = list(map(lambda step: '->'.join(step), steps))
        return np.asarray(steps)

def log_to_steps(log, cid_list, step_list, stepsz):
    # assumes all cases is at least stepsz length
    step_mapping = {step_list[i]:i for i in range(len(step_list))}
    rev_step_mapping = {val:key for key, val in step_mapping.items()}
    step_list = np.asarray(list(map(lambda step: step_mapping[tuple(step)], \
                                    step_list)))
    # create next step mapping for each step
    get_next_step = lambda step: list(filter(lambda step1: \
                                             rev_step_mapping[step1][0] == \
                                             rev_step_mapping[step][-1], \
                                             step_list))
    next_step_mapping = {step:get_next_step(step) for step in step_list}
    # convert log to step log dict
    step_log = dict()
    for cid in cid_list:
        case = log[(log['caseId']==cid)]['activity'].tolist()
        case = np.append([datrep.ARTIFICIAL_START,], case)

        step_case = np.asarray([(case[i - 1], case[i]) \
                                for i in range(1, len(case))])
        step_case = list(map(lambda step: step_mapping[tuple(step)], \
                             step_case))
        step_case = np.asarray(step_case)
        step_log[cid] = step_case

    return step_log, step_list, step_mapping, rev_step_mapping, \
            next_step_mapping
'''
def get_next_step(step, activity_list):
    # get steps whose first activity is the step's last activity
    # remove brackets
    acts = step.split('->')
    last_act = acts[-1]
    next_step_list = map(lambda a: last_act + '->' + a, activity_list)
    next_step_list = np.asarray(list(next_step_list))
    return next_step_list
'''

def convert_steplist(steplist, step_mapping):
    converted = map(lambda step: step_mapping[step], steplist)
    return np.asarray(list(converted))


def log_to_activity_steps(log, cid_list, activity_list, stepsz):
#    print('activity list:\n{}'.format(activity_list))
    step_list = pmutils.mk_possible_steps(activity_list, \
                                          stepsz=stepsz)
    print('Number of steps: {}'.format(len(step_list)))
    step_mapping = {step_list[i]:i for i in range(len(step_list))}
    rev_step_mapping = {val:key for key, val in step_mapping.items()}
    step_list = np.asarray(list(map(lambda step: step_mapping[tuple(step)], \
                                    step_list)))
    # create next step mapping for each step
    get_next_step = lambda step: list(filter(lambda step1: \
                                             rev_step_mapping[step1][0] == \
                                             rev_step_mapping[step][-1], \
                                             step_list))
    next_step_mapping = {step:get_next_step(step) for step in step_list}
   # convert log to step log dict
    step_log = dict()
    for cid in cid_list:
        case = log[(log['caseId']==cid)]['activity'].tolist()
        case = np.append([datrep.ARTIFICIAL_START,], case)
        step_case = np.asarray([(case[i - 1], case[i]) \
                                for i in range(1, len(case))])
        step_case = map(lambda step: step_mapping[tuple(step)], step_case)
        step_case = np.asarray(list(step_case))
        step_log[cid] = step_case

    return step_log, step_list, step_mapping, rev_step_mapping, \
            next_step_mapping


def step_log_to_variant(step_log):
    '''
    Map each case to a unique variant such that cases that are exactly the same
    belongs to the same variant.
    '''
    var_to_cid = dict()
    var_to_step = dict()
    step_to_var = dict()
    seen = set()
    vid = 0
    for cid, steps in step_log.items():
        steps_str = ','.join(steps.astype(np.str))
        if steps_str in seen:
            # already seen, get its variant
            variant_id = step_to_var[steps_str]
            # add to var_to_cid
            var_to_cid[variant_id].append(cid)
        else:
            # not seen
            seen.add(steps_str)
            # add to var_to_cid and var_to_step
            var_to_cid[vid] = list()
            var_to_cid[vid].append(cid)
            var_to_step[vid] = steps
            step_to_var[steps_str] = vid
            # update vid
            vid += 1
    return var_to_cid, var_to_step


if __name__ == '__main__':
    # load the small bpic2012 dataset
    log_small = './dataset/bpic2012/bpic2012FirstHundred.csv'
    log = pd.read_csv(log_small, index_col=False)
    cid_list = log['caseId'].unique().tolist()
    activity_list = log['activity'].unique().tolist()
    activity_list = np.append([datrep.ARTIFICIAL_START,], activity_list)
    activity_list = sorted(activity_list)

    stepsz = 2
    # convert log to activity steps
    start_time = time.time()
    step_log, step_list, step_mapping, \
            rev_step_mapping, next_step_mapping = \
            log_to_activity_steps(log, cid_list, activity_list, stepsz)
    end_time = time.time()

    print('Finished making small activity step log in {} seconds!' \
            .format(end_time - start_time))

    # try making big activity log
    log_full_dir = './dataset/bpic2012/bpic2012Full.csv'
    log_full = pd.read_csv(log_full_dir, index_col=False)
    cid_list = log_full['caseId'].unique().tolist()
    activity_list = log_full['activity'].unique().tolist()
    activity_list = np.append([datrep.ARTIFICIAL_START,], activity_list)
    activity_list = sorted(activity_list)
    
    # convert log to activity steps
    start_time = time.time()
    step_log, step_list, step_mapping, \
            rev_step_mapping, next_step_mapping = \
            log_to_activity_steps(log_full, cid_list, activity_list, stepsz)
    diff = time.time() - start_time

    print('Finished making full activity step log in {} seconds!'.format(diff))

    print('Getting variants of log...')
    var_to_cid, var_to_step = step_log_to_variant(step_log)
    print('Number of variants: {}'.format(len(var_to_cid)))

