#!/usr/bin/env python

import numpy as np
import pandas as pd
import os, sys, time
import dataRepresentation as datrep
import pmRecUtils as pmutils
import logUtils as lutils
from scipy.sparse import csc_matrix
from fastFM import als, sgd, mcmc
from sklearn.metrics import mean_squared_error


def setup(log_dir, column_dtypes, activity_classifier=('activity',)):
    log = pd.read_csv(log_dir, index_col=False, dtype=column_dtypes)
    cid_list = log['caseId'].unique().tolist()
    # assumes that the log is already sorted chronologically
    if len(activity_classifier) > 1:
        # print('Activity classifier: {}'.format(activity_classifier))
        # need to join columns to create concatenated activities
        # get the list of possible values for each classifier
        col_vals = log[activity_classifier].values
        new_act_col = list(map(lambda vals: '+'.join(vals), col_vals))
        # print('New act col: {}'.format(new_act_col))
        # rename the original activity column
        log.rename(index=str, columns={'activity':'activity_orig'}, \
                   inplace=True)
        log['activity'] = new_act_col
    activity_list = log['activity'].unique().tolist()
    activity_list = np.append([datrep.ARTIFICIAL_START,], activity_list)
    # print('Number of activities: {}'.format(len(activity_list)))
    # print('Activity list: \n{}'.format(activity_list))
    # sort alphabetically
    activity_list = np.asarray(sorted(activity_list))
    return log, cid_list, activity_list


def train_test_split(cid_list, train_perc=.7):
    length = len(cid_list)
    return cid_list[:int(length * train_perc)], \
            cid_list[int(length * train_perc):]


def train_valid_test_split(cid_list, train_perc=.6, valid_perc=.2):
    length = len(cid_list)
    train_ind = int(length * train_perc)
    valid_ind = train_ind + int(length * valid_perc)

    train_set = cid_list[:train_ind]
    valid_set = cid_list[train_ind:valid_ind]
    test_set = cid_list[valid_ind:]
    return train_set, valid_set, test_set


def _build_als_model(param):
    return als.FMRegression(n_iter=param['n_iter'], \
                            init_stdev=param['init_stdev'], \
                            rank=param['rank'], \
                            random_state=param['random_state'], \
                            l2_reg_w=param['l2_reg_w'], \
                            l2_reg_V=param['l2_reg_V'], \
                            l2_reg=param['l2_reg'])


def _build_sgd_model(param):
    return sgd.FMRegression(n_iter=param['n_iter'], \
                            init_stdev=param['init_stdev'], \
                            rank=param['rank'], \
                            random_state=param['random_state'], \
                            l2_reg_w=param['l2_reg_w'], \
                            l2_reg_V=param['l2_reg_V'], \
                            l2_reg=param['l2_reg'], \
                            step_size=param['step_size'])


def _build_mcmc_model(param):
    return mcmc.FMRegression(n_iter=param['n_iter'], \
                             init_stdev=param['init_stdev'], \
                             rank=param['rank'], \
                             random_state=param['random_state'])


def build_model(model_type, param):
    model_map = {'als':_build_als_model, 'sgd':_build_sgd_model, \
                 'mcmc':_build_mcmc_model}
    fm_model = model_map[model_type](param)
    return fm_model


def run_experiment(train_X, train_y, test_X, fm_model):
    # train model
    start = time.time()
    fm_model.fit(train_X, train_y)
    diff_train = time.time() - start
    print('Time taken to train model: {} seconds'.format(diff_train))
    # make predictions
    start = time.time()
    y_pred = fm_model.predict(test_X)
    diff_test = time.time() - start
    print('Time taken to make predictions: {} seconds.'.format(diff_test))
    time_df = pd.DataFrame({'train':[diff_train,], 'test':[diff_test,]})
    return y_pred, time_df


def run_optimal(train_X, train_y, test_X, test_y, fm_model, \
                           max_iter, step_size=1, stop_delta=.2):
    start = time.time()
    fm_model.fit(train_X, train_y)

    rmse_train_trace = []
    rmse_test_trace = []

    rmse_min = np.inf
    it_rmse_min = -1
    best_y_pred = None
    last_rmse = None
    it = 0
    for i in range(1, max_iter):
        if i // 1000 != it:
            print('Learning at iteration: {} / {}'.format(i, max_iter))
            print('Current best test rmse: {} at it: {}'.format(rmse_min, \
                                                                it_rmse_min))
            print('Last rmse: {} at it: {}'.format(last_rmse, i - 1))
            it = i // 1000

        fm_model.fit(train_X, train_y, n_more_iter=step_size)

        y_pred_train = fm_model.predict(train_X)
        y_pred = fm_model.predict(test_X)
        rmse_train = np.sqrt(mean_squared_error(y_pred_train, train_y))
        rmse_test = np.sqrt(mean_squared_error(y_pred, test_y))
        rmse_train_trace.append(rmse_train)
        rmse_test_trace.append(rmse_test)
        # update last rmse
        last_rmse = rmse_test

        # check for early stopping
        if rmse_min > rmse_test:
            # error dropping continue
            rmse_min = rmse_test
            best_y_pred = y_pred
            it_rmse_min = i
        elif rmse_test - rmse_min > stop_delta:
            # break for loop to early stop
            break
    diff = time.time() - start
    print('Time taken to run experiment: {} seconds.'.format(diff))
    time_df = pd.DataFrame({'train':[diff]})
    rmse_df = pd.DataFrame({'rmse_train':rmse_train_trace, \
                            'rmse_test':rmse_test_trace})
    return best_y_pred, time_df, rmse_df



def model_select(train_X, train_y, valid_X, valid_y, \
                 fm_model, max_iter, step_size=1, stop_delta=.2):
    start = time.time()
    fm_model.fit(train_X, train_y)

    rmse_train_trace = list()
    rmse_valid_trace = list()

    rmse_min = np.inf
    it_rmse_min = -1
    best_y_pred = None
    last_rmse = None
    it_counter = -1
    it = -1

    # do not stop while it is still under max iter or it is under 20% of
    # max_iter
    while it < max_iter - 1 or it < int(max_iter * .1):
        # update iteration
        it += 1

        if it == 1 or (it > 0 and it // 1000 != it_counter):
            print('Learning at iteration {} / {}'.format(it, max_iter))
            print('Current best validation rmse: {} at it: {}'\
                  .format(rmse_min, it_rmse_min))
            print('Last rmse: {} at it: {}'.format(last_rmse, it - 1))
            it_counter = it // 1000

        fm_model.fit(train_X, train_y, n_more_iter=step_size)

        y_pred_train = fm_model.predict(train_X)
        y_pred_valid = fm_model.predict(valid_X)
        rmse_train = np.sqrt(mean_squared_error(y_pred_train, train_y))
        rmse_valid = np.sqrt(mean_squared_error(y_pred_valid, valid_y))
        rmse_train_trace.append(rmse_train)
        rmse_valid_trace.append(rmse_valid)
        # update last rmse
        last_rmse = rmse_valid

        # check for early stopping
        if rmse_min > rmse_valid:
            # error dropping continue
            rmse_min = rmse_valid
            best_y_pred = y_pred_valid
            # iteration started with index 0
            it_rmse_min = (it + 1) * step_size
        elif rmse_valid > rmse_min and it > int(max_iter * .1):
            # break for loop to early stop if current rmse_valid is higher
            # and it is over min iterations
            break
#        elif rmse_valid - rmse_min > stop_delta and it > int(max_iter * .2):
#            # break for loop to early stop, needs to be over min iterations
#            break

    rmse_train_trace = np.asarray(rmse_train_trace)
    rmse_valid_trace = np.asarray(rmse_valid_trace)

    diff = time.time() - start
    print('Time taken model selection: {} seconds.'.format(diff))
    time_df = pd.DataFrame({'train':[diff]})
    rmse_df = pd.DataFrame({'rmse_train':rmse_train_trace[:it+1], \
                            'rmse_valid':rmse_valid_trace[:it+1]})
    return it_rmse_min, time_df, rmse_df


def export_results(top_pred_df, time_df, outdir):
    # save top predictions y_pred separately
    store_pred = pd.HDFStore(outdir + os.sep + 'topYPred.h5')
    top_pred_y_df = top_pred_df[['target', 'prediction']]
    store_pred['y_pred'] = top_pred_y_df
    store_pred.close()
    # save top prediction in a HDFStore
    store = pd.HDFStore(outdir + os.sep + 'topPred.h5')
    store['top_pred'] = top_pred_df
    store.close()
    # save time df in a separate store
    store_time = pd.HDFStore(outdir + os.sep + 'time.h5')
    store_time['time'] = time_df
    store_time.close()


def export_results_light(top_pred_df, time_df, outdir):
    store = pd.HDFStore(outdir + os.sep + 'topYPred.h5')
    store['y_pred'] = top_pred_df
    store.close()
    store_time = pd.HDFStore(outdir + os.sep + 'time.h5')
    store_time['time'] = time_df
    store_time.close()

'''
def export_results(train_X, train_y, test_X, test_y, prediction, \
                   pred_id_list, log_mat_colnames, time_df, outdir):
    # save everything in a HDFStore
    store = pd.HDFStore(outdir + os.sep + 'results.h5')
    store['time'] = time_df
    # turn csc matrices to sparse matrix
    train_X_df = pd.SparseDataFrame(train_X)
    train_X_df.columns = log_mat_colnames
    train_y_df = pd.DataFrame(train_y)
    train_y_df.columns = ['target',]
    test_X_df = pd.SparseDataFrame(test_X)
    test_X_df.columns = log_mat_colnames
    test_y_df = pd.DataFrame(test_y)
    test_y_df.columns = ['target',]
    prediction_df = pd.DataFrame(prediction)
    prediction_df.columns = ['prediction',]
    pred_id_df = pd.DataFrame(pred_id_list, dtype=np.int)
    pred_id_df.columns = ['pred_id']

    store['train_X'] = train_X_df
    store['train_y'] = train_y_df
    store['test_X'] = test_X_df
    store['test_y'] = test_y_df
    store['prediction'] = prediction_df
    store['pred_id'] = pred_id_df

    store.close()
'''


if __name__ == '__main__':
    # small experiment with small log
    log_small_dir = './dataset/bpic2012/bpic2012FirstHundred.csv'
    log_full_dir = './dataset/bpic2012/bpic2012Full.csv'
    
    # hyperparameters
    hyper = dict()
    hyper['stepsz'] = 2
    hyper['minpartialsz'] = 2
    hyper['negative_samples'] = 0
    hyper['seed'] = 123
    hyper['normalize'] = True

    log_small, cid_list, activity_list = setup(log_small_dir)
    # need the full log to ensure we get the full activity list
    _, _, activity_list = setup(log_full_dir)

    step_log, step_list, step_mapping, \
            rev_step_mapping, next_step_mapping = \
            lutils.log_to_activity_steps(log_small, cid_list, activity_list, \
                                         hyper['stepsz'])

    print('Finished!')






