#!/usr/bin/env python

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix


def pick_top_prediction(test_X, test_y, pred_id_list, pred, log_mat_colnames):
    print('Turning matrices to dataframes...')
    test_X_df = pd.SparseDataFrame(test_X)
    test_X_df.columns = log_mat_colnames
    test_y_df = pd.DataFrame(test_y)
    test_y_df.columns = ['target',]
    pred_id_df = pd.DataFrame(pred_id_list, dtype=np.int)
    pred_id_df.columns = ['pred_id',]
    pred_df = pd.DataFrame(pred, dtype=np.float64)
    pred_df.columns = ['prediction',]
    print('Finished.')

    # join them
    joined_df = pd.concat([test_X_df, test_y_df, \
                           pred_id_df, pred_df], axis=1).to_sparse()
    print('Joined df type: {}'.format(type(joined_df)))
    print('Sorting joined df...') 
    joined_df_sorted = joined_df.sort_values('prediction', ascending=False, \
                                             inplace=False, kind='mergesort')
    print('Finished sorting.')
    max_pid = int(np.max(pred_id_list))
    print('Picking {} top predictions...'.format(max_pid))
    top_preds_df = joined_df_sorted.groupby('pred_id', sort=False)\
            .head(1).reset_index(drop=True)
    print('Finished picking.')
    #    print('top_preds_df type: {}'.format(type(top_preds_df)))
#    print('top_preds_df cols: {}'.format(top_preds_df.columns))
#    print('top_preds_df index: {}'.format(top_preds_df.index))
    assert not top_preds_df.empty
    return top_preds_df


def pick_top_prediction_light(test_y, pred_id_list, pred):
    print('Turning matrices to dataframes...')
    test_y_df = pd.SparseDataFrame(test_y, dtype=np.int, columns=['target'])
    pred_id_df = pd.SparseDataFrame(pred_id_list,  dtype=np.int, \
                                    columns=['pred_id'])
    pred_df = pd.SparseDataFrame(pred, dtype=np.float64, \
                                 columns=['prediction'])
    print('Finished.')

    # join them
    joined_df = pd.concat([test_y_df, pred_id_df, pred_df], axis=1).to_sparse()
#    print('Joined df type: {}'.format(type(joined_df)))
    print('Sorting joined df...')
    joined_df_sorted = joined_df.sort_values('prediction', ascending=False, \
                                             inplace=False, kind='mergesort')
    print('Finished sorting.')
    max_pid = int(np.max(pred_id_list))
    print('Picking {} top predictions...'.format(max_pid))
    top_preds_df = joined_df_sorted.groupby('pred_id', sort=False)\
            .head(1).reset_index(drop=True)
    print('Finished picking.')
    #    print('top_preds_df type: {}'.format(type(top_preds_df)))
#    print('top_preds_df cols: {}'.format(top_preds_df.columns))
#    print('top_preds_df index: {}'.format(top_preds_df.index))
    assert not top_preds_df.empty
    return top_preds_df


def rank_top_prediction(test_y, pred_id_list, pred_list):
    print('Turning matrices to dataframes...')
    test_y_df = pd.SparseDataFrame(test_y, dtype=np.int, columns=['target'])
    pred_id_df = pd.SparseDataFrame(pred_id_list, dtype=np.int, \
                                    columns=['pred_id'])
    pred_df = pd.SparseDataFrame(pred_list, dtype=np.float64, \
                                 columns=['prediction'])
    print('Finished.')

    # join them
    joined_df = pd.concat([test_y_df, pred_id_df, pred_df], axis=1).to_sparse()
    print('Sorting joined df...')
    joined_df_sorted = joined_df.sort_values('prediction', ascending=False, \
                                             inplace=False, kind='mergesort')
#    print('joined_df_sorted: \n{}'.format(joined_df_sorted))
    max_pid = int(np.max(pred_id_list))
    print('Picking {} top predictions...'.format(max_pid))
    top_preds_df = joined_df_sorted.groupby('pred_id', sort=False)
    top_preds_df = top_preds_df.head(1).reset_index(drop=True)

    return top_preds_df


def rank_top_prediction_dense(test_y, pred_id_list, pred_list):
    print('Turning matrices to dataframes...')
    test_y_df = pd.DataFrame(test_y, dtype=np.int, columns=['target'])
    pred_id_df = pd.DataFrame(pred_id_list, dtype=np.int, \
                                    columns=['pred_id'])
    pred_df = pd.DataFrame(pred_list, dtype=np.float64, \
                                 columns=['prediction'])
    assert test_y_df.shape[0] == pred_id_df.shape[0] and \
            pred_id_df.shape[0] == pred_df.shape[0]
    print('Finished.')

    # join them
    joined_df = pd.concat([test_y_df, pred_id_df, pred_df], axis=1)
    print('Sorting joined df...')
    joined_df_sorted = joined_df.sort_values('prediction', ascending=False, \
                                             inplace=False, kind='mergesort')
#    print('joined_df_sorted: \n{}'.format(joined_df_sorted))
    max_pid = int(np.max(pred_id_list))
    print('Picking {} top predictions...'.format(max_pid))
    top_preds_df = joined_df_sorted.groupby('pred_id', sort=False)
    top_preds_df = top_preds_df.head(1).reset_index(drop=True)

    return top_preds_df

'''
def pick_top_prediction(test_X_df, test_y_df, pred_id_df, pred_df):
    # Returns the top prediction rows

    joined_df = pd.concat([test_X_df, test_y_df, \
                                pred_id_df, pred_df], axis=1)
    # sort the df by prediction
    joined_df_sorted = joined_df.sort_values('prediction', ascending=False, \
                                             inplace=False) 
    top_preds = list()
    for pid in joined_df_sorted['pred_id'].unique().tolist():
        sub_df = joined_df_sorted[(joined_df_sorted['pred_id']==pid)]
        # get the top row, this will be the top prediction
#        print('sub df is empty for pid {}: {}'.format(pid, sub_df.empty))
        top_pred = sub_df.iloc[0,:]
        top_preds.append(top_pred.values)

    # make into dataframe
    top_preds_df = pd.DataFrame(top_preds)
    top_preds_df.columns = joined_df_sorted.columns
    return top_preds_df
'''
