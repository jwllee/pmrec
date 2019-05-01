#!/usr/bin/env python

import pandas as pd
import numpy as np
import os, sys, time, sklearn
from sklearn.model_selection import train_test_split
import pmRecUtils as rutils
import dataRepresentation as datrep
from scipy.sparse import csc_matrix
import itertools as itools
import fmExperiment as fmexp
import fmEvaluation as fmeval
from dataRep import rStep
import utils 


from fastFM import als, sgd, mcmc


logger = utils.make_logger(utils.MINIMAL_DEBUG)


ARTIFICIAL_START = '0'
RANDOM_STATE = 1


def run(log_fp):
    logger.info('Reading log...')
    log_df = pd.read_csv(log_fp, index_col=False)
    log_df['activity_label'] = log_df['activity']
    log_df['activity'] = log_df['activity_id'].astype(str)
    log_df = log_df[['caseId', 'activity']]
    
    logger.debug('log_df shape: {}'.format(log_df.shape))
    logger.debug(log_df.head())

    min_caselen = 4
    logger.info('Filter log to include cases of length > {}'.format(min_caselen))
    caselen_df = log_df.groupby('caseId', as_index=False).count()
    short_cases = caselen_df['activity'] <= min_caselen
    cid_to_excl = caselen_df.loc[short_cases, 'caseId']
    log_df = log_df.loc[~(log_df['caseId'].isin(cid_to_excl)), :]

    logger.debug('Filtered out {} cases'.format(short_cases.sum()))
    logger.debug('log_df shape: {}'.format(log_df.shape))

    logger.info('Creating itemlists...')
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

    train_size = 0.7
    logger.info('Train test split with train size: {}...'.format(train_size))
    train_cid_list, test_cid_list = train_test_split(cid_list, 
                                                     train_size=train_size, 
                                                     random_state=RANDOM_STATE)

    train_df = log_df.loc[(log_df['caseId'].isin(train_cid_list)), :]
    test_df = log_df.loc[(log_df['caseId'].isin(test_cid_list)), :]

    logger.info('Build train FM csc_matrix...')
    train_formatted = rStep.log_to_fm_format(train_df, stepsz, cid_list,
                                             step_dict, next_step_mapping,
                                             negative_samples=3)

    train_x_datalist = train_formatted[0]
    train_x_row_inds = train_formatted[1]
    train_x_col_inds = train_formatted[2]
    train_x_shape = train_formatted[3]
    train_y_datalist = train_formatted[4]
    train_pred_id_list = train_formatted[5]

    train_X = csc_matrix((train_x_datalist, (train_x_row_inds,
                                             train_x_col_inds)), train_x_shape)
    train_y = train_y_datalist

    logger.info('Build test FM csc_matrix...')
    test_formatted = rStep.log_to_fm_format(test_df, stepsz, cid_list,
                                            step_dict, next_step_mapping,
                                            negative_samples=-1)
    
    test_x_datalist = test_formatted[0]
    test_x_row_inds = test_formatted[1]
    test_x_col_inds = test_formatted[2]
    test_x_shape = test_formatted[3]
    test_y_datalist = test_formatted[4]
    test_pred_id_list = test_formatted[5]
    test_y_label = test_formatted[6]

    test_X = csc_matrix((test_x_datalist, (test_x_row_inds,
                                           test_x_col_inds)), test_x_shape)
    test_y = test_y_datalist

    logger.info('Build FM model...')

    n_iter = 10000
    init_stdev = .01
    rank = 8
    l2_reg_w = .01
    l2_reg_V = .01
    l2_reg = 0
    step_size = .01

    fm_model = sgd.FMRegression(n_iter=n_iter,
                                init_stdev=init_stdev,
                                rank=rank, random_state=RANDOM_STATE,
                                l2_reg_w=l2_reg_w,
                                l2_reg_V=l2_reg_V,
                                step_size=step_size)

    logger.info('Training...')
    train_start = time.time()
    fm_model.fit(train_X, train_y)
    train_took = time.time() - train_start 
    logger.info('Took {:.2f}s to train'.format(train_took))

    logger.info('Predicting...')
    pred_y = fm_model.predict(test_X)
    pred_df = pd.DataFrame({
        'pred_id': test_pred_id_list,
        'step_label': test_y_label,
        'pred_y': pred_y,
        'test_y': test_y,
    })
    logger.debug('Predictions: \n{}'.format(pred_df.head()))
    
    grouped = pred_df.groupby('pred_id', as_index=False)
    k = 1
    top_k_pred_df = grouped.apply(lambda df: df.nlargest(k, columns='pred_y'))
    logger.debug('Top {} predictions: \n{}'.format(k, top_k_pred_df.head()))
    acc = top_k_pred_df['test_y'].sum() / top_k_pred_df.shape[0] * 100.
    logger.debug('Accuracy: {:.2f}%'.format(acc))


if __name__ == '__main__':
    log_name = 'BPIC2012-len_100.csv'
    log_fp = os.path.join('..', '..', 'data', log_name)

    run(log_fp)
