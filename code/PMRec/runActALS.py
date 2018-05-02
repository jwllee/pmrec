#!/usr/bin/env python

import pandas as pd
import numpy as np
import os, sys, time
import pmRecUtils as rutils
import dataRepresentation as datrep
from scipy.sparse import csc_matrix
import itertools as itools
import fmExperiment as fmexp
import fmEvaluation as fmeval


def run(param, data_dict=None):
    if data_dict == None:
        has_data = False
        data_dict = dict()
    else:
        has_data = True

    # write params
    param_df =  pd.DataFrame(param, index=[0])
    param_df = param_df[sorted(param_df.columns.values)]
    param_df.to_csv(param['result_dir'] + os.sep + 'params.csv', index=False)

    # read log
    print('Reading log...')
    if not has_data:
        log = pd.read_csv(param['log_dir'], index_col=False, \
                          dtype=param['column_dtypes'])
        # filter by minimum case lengths
        excl_cids = log[['caseId', 'activity']]
        excl_cids = excl_cids.groupby('caseId', as_index=False).count()
        min_len = param['min_case_len']
        excl_cids = excl_cids[(excl_cids['activity']<=min_len)]
        excl_cids = excl_cids.loc[:,['caseId']]
        log = log[~(log['caseId'].isin(excl_cids))]
        data_dict['log'] = log
        data_dict['excl_cids'] = excl_cids
    else:
        log = data_dict['log']
        excl_cids = data_dict['excl_cids']

    excl_cids.to_csv(param['result_dir'] + os.sep + 'excluded.csv', \
                 index=False)

    # creating all the itemlists
    print('Creating itemlists...')
    if not has_data:
        # activity list
        activity_list = log['activity'].unique().tolist()
        activity_list = sorted(activity_list)
        activity_list = np.asarray(activity_list)
        # caseid list
        cid_list = log['caseId'].unique()
        cid_list.sort()

        data_dict['activity_list'] = activity_list
        data_dict['cid_list'] = cid_list
    else:
        activity_list = data_dict['activity_list']
        cid_list = data_dict['cid_list']

    if param['feature'] == 'only_setup':
        return data_dict

    print('Splitting train, valid and test set...')
    if not has_data:
        train_cid_list, valid_cid_list, test_cid_list = \
                fmexp.train_valid_test_split(cid_list, param['train_perc'], \
                                             param['valid_perc'])
        data_dict['train_cid_list'] = train_cid_list
        data_dict['valid_cid_list'] = valid_cid_list
        data_dict['test_cid_list'] = test_cid_list
    else:
        train_cid_list = data_dict['train_cid_list']
        valid_cid_list = data_dict['valid_cid_list']
        test_cid_list = data_dict['test_cid_list']

    print('Building train matrix...')
    time_dict = dict()
    time_dict['build_train'] = time.time()
    if not has_data:
        train_log = log[(log['caseId'].isin(train_cid_list))]
        train_X_datalist, train_X_row_inds, train_X_col_inds, \
                train_X_shape, train_y_datalist, id_list = \
                datrep.step_log_to_fm_format_all(param['feature'], \
                                                 train_log, cid_list, \
                                                 activity_list, \
                                                 param['minpartialsz'], \
                                                 param['negative_samples'], \
                                                 param['seed'], \
                                                 param['normalize'])
        print('train_X shape: {}'.format(train_X_shape))
        train_X = csc_matrix((train_X_datalist, (train_X_row_inds, \
                                                 train_X_col_inds)), train_X_shape)
        train_y = train_y_datalist

        data_dict['train_X'] = train_X
        data_dict['train_y'] = train_y
        sparsity_dict = dict()
        train_sparsity = len(train_X_datalist) * 1. / train_X_shape.prod()
        sparsity_dict['train'] = train_sparsity
        data_dict['sparsity'] = sparsity_dict
    else:
        train_X = data_dict['train_X']
        train_y = data_dict['train_y']
        sparsity_dict = data_dict['sparsity']

    time_dict['build_train'] = time.time() - time_dict['build_train']
    print('Took {} seconds to build train matrix.'\
          .format(time_dict['build_train']))
    print('Train matrix shape: {}'.format(train_X.shape))

    print('Building valid matrix...')
    time_dict['build_valid'] = time.time()
    if not has_data:
        valid_log = log[(log['caseId'].isin(valid_cid_list))]
        valid_X_datalist, valid_X_row_inds, valid_X_col_inds, \
                valid_X_shape, valid_y_datalist, _ = \
                datrep.step_log_to_fm_format_all(param['feature'], \
                                                 valid_log, cid_list, \
                                                 activity_list, \
                                                 param['minpartialsz'], \
                                                 negative_samples=-1, \
                                                 seed=param['seed'], \
                                                 normalize=param['normalize'])

        valid_X = csc_matrix((valid_X_datalist, (valid_X_row_inds, \
                                               valid_X_col_inds)), valid_X_shape)
        valid_y = valid_y_datalist

        data_dict['valid_X'] = valid_X
        data_dict['valid_y'] = valid_y
        valid_sparsity = len(valid_X_datalist) * 1. / valid_X_shape.prod()
        sparsity_dict['valid'] = valid_sparsity
    else:
        valid_X = data_dict['valid_X']
        valid_y = data_dict['valid_y']

    time_dict['build_valid'] = time.time() - time_dict['build_valid']
    print('Took {} seconds to build valid matrix.'\
          .format(time_dict['build_valid']))
    print('Valid matrix shape: {}'.format(valid_X.shape))

    print('Building test matrix...')
    time_dict['build_test'] = time.time()
    if not has_data:
        test_log = log[(log['caseId'].isin(test_cid_list))]
        test_X_datalist, test_X_row_inds, test_X_col_inds, \
                test_X_shape, test_y_datalist, pred_id_list = \
                datrep.step_log_to_fm_format_all(param['feature'], \
                                                 test_log, cid_list, \
                                                 activity_list, \
                                                 param['minpartialsz'], \
                                                 negative_samples=-1, \
                                                 seed=param['seed'], \
                                                 normalize=param['normalize'])

        test_X = csc_matrix((test_X_datalist, (test_X_row_inds, \
                                               test_X_col_inds)), test_X_shape)
        test_y = test_y_datalist

        data_dict['test_X'] = test_X
        data_dict['test_y'] = test_y
        data_dict['pred_id_list'] = pred_id_list
        test_sparsity = len(test_X_datalist) * 1. / test_X_shape.prod()
        sparsity_dict['test'] = test_sparsity
    else:
        test_X = data_dict['test_X']
        test_y = data_dict['test_y']
        pred_id_list = data_dict['pred_id_list']

    time_dict['build_test'] = time.time() - time_dict['build_test']
    print('Took {} seconds to build test matrix.'\
          .format(time_dict['build_test']))
    print('Test matrix shape: {}'.format(test_X.shape))

    # build fm model
    fm_model = fmexp.build_model(param['model_type'], param)
    # model selection on number of iterations
    best_it, time_df, rmse_df = fmexp.model_select(train_X, train_y, \
                                                   valid_X, valid_y, \
                                                   fm_model, \
                                                   param['max_iter'], \
                                                   param['step_size_als'], \
                                                   param['stop_delta'])

    print('Best n_iter: {}'.format(best_it))
    # retrain fm model to the optimal num of iterations
    print('Retraining model for prediction...')
    time_dict['retrain'] = time.time()
    fm_model = fmexp.build_model(param['model_type'], param)
    fm_model.fit(train_X, train_y)
    fm_model.fit(train_X, train_y, n_more_iter=best_it)
    time_dict['retrain'] = time.time() - time_dict['retrain']
    print('Took {} seconds to retrain model.'\
          .format(time_dict['retrain']))

    print('Making predictions...')
#    y_pred, time_df = fmexp.run_experiment(train_X, train_y, test_X, fm_model)
    time_dict['test'] = time.time()
    y_pred = fm_model.predict(test_X)
    time_dict['test'] = time.time() - time_dict['test']
    print('Finished predictions!')

    for key, item in time_dict.items():
        time_df[key] = item

    print('Ranking to get the top predictions...')
    time_dict['picking_preds'] = time.time()
    top_pred_df = fmeval.rank_top_prediction_dense(test_y, pred_id_list, y_pred)
    time_dict['picking_preds'] = time.time() - time_dict['picking_preds']
    print('Took {} seconds to pick top predictions.'\
          .format(time_dict['picking_preds']))

    start_export = time.time()
#    fmexp.export_results_light(top_pred_df, time_df, result_dir)
    top_pred_df.to_csv(result_dir + os.sep + 'topYPred.csv', index=False)
    time_df = time_df[sorted(time_df.columns)]
    time_df.to_csv(result_dir + os.sep + 'time.csv', index=False)
    rmse_df.to_csv(result_dir + os.sep + 'learnRate.csv', index=False)
    sparsity_df = pd.DataFrame(sparsity_dict, index=[0])
    sparsity_df = sparsity_df[sorted(sparsity_df.columns)]
    sparsity_df.to_csv(result_dir + os.sep + 'sparsity.csv', index=False)
    # export train, valid, and test cid list
    train_cid_df = pd.DataFrame({'caseId':train_cid_list})
    valid_cid_df = pd.DataFrame({'caseId':valid_cid_list})
    test_cid_df = pd.DataFrame({'caseId':test_cid_list})
    train_cid_df.to_csv(result_dir + os.sep + 'trainCidList.csv', index=False)
    valid_cid_df.to_csv(result_dir + os.sep + 'validCidList.csv', index=False)
    test_cid_df.to_csv(result_dir + os.sep + 'testCidList.csv', index=False)
    diff_export = time.time() - start_export
    print('Took {} seconds to export results.'.format(diff_export))

    return data_dict




if __name__ == '__main__':
    # run experiment using different params
    data_dict = None
    # different configurations
    feature_list = ['act']
    n_iter_list = [0]
    rank_list = [20, 50, 100, 150]
    init_stdev_list = [.1]
    l2_reg_w_list = [.5]
    l2_reg_V_list = [.1]
    l2_reg_list = [0]
    # ALS configurations
    max_iter = 10000
    step_size_als = 1
    stop_delta = .1
    log_dirs = ['FirstTen', 'Complete', 'Full', 'WComplete', 'WFull', 'AFull', 'OFull']
    log_dirs = ['OFull']
    log_dirs = map(lambda d: '..' + os.sep + 'dataset' + os.sep + \
                    'bpic2012' + os.sep + 'bpic2012' + d + '.csv', log_dirs)
    log_dirs = list(log_dirs)
    negative_sample_list = [3, 0, 5, -1]
    configs_major = itools.product(log_dirs, negative_sample_list)
    for log_dir, negative_sample in configs_major:
        print('Running experiment on {}, negative sample: {}'\
              .format(log_dir, negative_sample))
        configs_minor = itools.product(feature_list, rank_list, n_iter_list, \
                                      init_stdev_list, l2_reg_w_list, \
                                      l2_reg_V_list, l2_reg_list)
        for feature, rank, n_iter, init_stdev, l2_reg_w, \
            l2_reg_V, l2_reg in configs_minor:
            param = dict()
            param['minpartialsz'] = 4
            param['negative_samples'] = negative_sample
            param['seed'] = 123
            param['normalize'] = True
            param['train_perc'] = .6
            param['valid_perc'] = .2
            param['test_perc'] = .2
            param['feature'] = feature
            param['log_dir'] = log_dir
            # parameters for the fm model
            param['model_type'] = 'als'
            param['n_iter'] = n_iter
            param['init_stdev'] = init_stdev
            param['rank'] = rank
            param['random_state'] = 123
            param['l2_reg_w'] = l2_reg_w
            param['l2_reg_V'] = l2_reg_V
            param['l2_reg'] = l2_reg
            param['min_case_len'] = 3
            # parameters for als fm model
            param['max_iter'] = max_iter
            param['step_size_als'] = step_size_als
            param['stop_delta'] = stop_delta
            param['column_dtypes'] = {'caseId':np.long, 'activity':str, 'timestamp':np.long, \
                                      'lifecycle':str, 'resource':str, \
                                      'amount_req':np.float64}

            # create result dir
            base_dir = os.path.split(os.getcwd())[0] + os.sep + 'out'
            base_dir = '..' + os.sep + '..' + os.sep + 'out'
            result_dir = base_dir + os.sep + 'actOFullALS' + os.sep + \
                    time.strftime('%d-%m-%Y_%H-%M-%S')
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
            param['result_dir'] = result_dir

            print('Running experiment for {} n_iter'.format(n_iter))
            data_dict = run(param, data_dict)
            time.sleep(1)
        data_dict = None

