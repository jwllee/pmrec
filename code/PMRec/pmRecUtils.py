#!/usr/bin/env python

import itertools as itools
import numpy as np
import pandas as pd
import functools as fct
import os, sys, time
import logUtils as lutils


def single_to_fm_format(data, itemlist):
    '''Convert single entry data (each entry consists of one item in data) to required
    format for fastFM. Creates the needed input (data, (row, column)) format for csc matrix
    with dimension (|data| x |itemlist|).

    :param data: single entry data
    :type data: narray
    :param itemlist: ordered list of possible item values
    :type itemlist: narray

    :return: data, row indexes, column indexes, shape
    :rtype: tuple
    '''
    column = len(itemlist)
    row = len(data)
    shape = (row, column)

    '''
    # numpy approach actually slower than using for loop...
    datalist = np.ones(len(data))
    row_inds = np.asarray(range(len(data)))
    # sort the itemlist
    xsorted = np.argsort(itemlist)
    # get the sorted itemlist
    sorted_itemlist = itemlist[xsorted]
    # get the indices of each data in data at the sorted_itemlist
    ypos = np.searchsorted(sorted_itemlist, data)
    # get the indices of the unsorted itemlist using the sorted indices
    # raises error if some data is not in itemlist
    col_inds = xsorted[ypos]

    '''
    row_inds = np.zeros(shape=(len(data),), dtype=np.int)
    col_inds = np.zeros(shape=(len(data),), dtype=np.int)
    datalist = np.zeros(shape=(len(data),), dtype=np.int)

    for i in range(len(data)):
        item = data[i]
        # this is the cell value for entry item
        val = 1
        datalist[i] = val
        # locate its position in the itemlist, throws error if item is not a
        # possible item
        col_ind = np.where(itemlist==item)[0]
#        print('col_ind: {}'.format(col_ind))
        # should not be duplicated items in the itemlist
        assert len(col_ind) == 1
        col_ind = col_ind[0]
        row_ind = i

        col_inds[i] = col_ind
        row_inds[i] = row_ind

    return datalist, row_inds, col_inds, shape


def multiple_to_fm_format(data, itemlist, normalize=True):
    '''Convert multi-entry data (each entry consists of one or more items in data) to required
    format for fastFM. Creates the needed input (data, (row, column)) format for csc matrix 
    with dimension (|data| x |itemlist|)

    :param data: multi-entry data
    :type data: a multidimension ndarray
    :param itemlist: ordered list of possible item values
    :type itemlist: ndarray
    :param normalize: whether to normalize the entry row so that it sums to 1
    :type normalize: bool

    :return: datalist, row indexes, column indexes, shape
    :rtype: tuple
    '''
    column = len(itemlist)
    row = len(data)
    shape = (row, column)

    # number of data
    num_of_data = fct.reduce(lambda x, y: x + len(y), data, 0)
    row_inds = np.zeros(num_of_data, dtype=np.int)
    col_inds = np.zeros(num_of_data, dtype=np.int)
    datalist = np.zeros(num_of_data, dtype=np.float64)
    cnt = 0
    for i in range(len(data)):
        multi_entry = data[i]

        if normalize:
            val = 1. / len(multi_entry)
        else:
            val = 1.

        # for each item in multi_entry, locate its position in the itemlist
        # throws error if item is not a possible item
        # all the items stay at the same row
        row_ind = i
        col_ind_list = list(map(lambda item: np.where(itemlist==item)[0], multi_entry))

        assert len(col_ind_list) == len(multi_entry)

        '''
        start = time.time()
        if not np.asarray(list(map(lambda l: len(l)>0, col_ind_list))).all():
            print('Multi entry: {}'.format(multi_entry))
            col_ind_list_1 = list(map(lambda l: list(l), col_ind_list))
            print('col_ind_list: {}'.format(col_ind_list_1))
        diff = time.time() - start
        print('Time taken to check: {}'.format(diff))
        '''

        '''
        # indices to replace
        to_set = np.arange(len(col_ind_list)) + cnt
        vals = np.ones(len(col_ind_list), np.float64) * val
        datalist.put(to_set, vals)
        col_inds.put(to_set, col_ind_list)
        row_inds_i = np.ones(len(col_ind_list)) * row_ind
        row_inds.put(to_set, row_inds_i)
        # update cnt
        cnt = to_set[-1] + 1
        '''
        for i in range(len(col_ind_list)):
            col_ind = col_ind_list[i]
            assert col_ind.shape[0] == 1, 'col_ind: {}'.format(col_ind)
            datalist[cnt] = val
#            print('col_ind: {}, type: {}'.format(col_ind, type(col_ind)))
            col_inds[cnt] = col_ind
            row_inds[cnt] = row_ind

            # update count
            cnt += 1

    return datalist, row_inds, col_inds, shape


def mk_possible_steps(activities, stepsz=2):
    '''Create a dataframe of possible activity steps

    :param activities: list of activities
    :type activities: list
    :param stepsz: number of activities per step
    :type stepsz: int
    :return: activity steps
    :rtype: DataFrame
    '''

    # create permutations of activities as steps
    steps = list(itools.product(activities, repeat=stepsz))
    steps = list(map(lambda step: '->'.join(step), steps))

    return np.asarray(steps)


def mk_step_block(case, step_dict, stepsz, repeat=1, normalize=True):
    assert stepsz >= 2

    datalist = list()
    row_inds = list()
    col_inds = list()
    shape = np.zeros(2)

    if case.shape[0] == 0:
        return np.asarray(datalist), np.asarray(row_inds), \
                np.asarray(col_inds), shape

    for sz in range(2, stepsz + 1):
        if case.shape[0] < sz:
            continue

        steps = lutils.case_to_steps(case, sz)
        steps_repeat = [steps for _ in range(repeat)]
        step_list = step_dict[sz]
        datalist_i, row_inds_i, col_inds_i, shape_i = \
                multiple_to_fm_format(steps_repeat, step_list, normalize)
        # shift results if needed
        if (shape == np.zeros(2)).all():
            shape = shape_i
        else:
            to_shift = step_dict[sz - 1].shape[0]
            col_inds_i += to_shift
            assert shape[0] == shape_i[0]
#            print('shape: {}, shape_i: {}'.format(shape, shape_i))
            shape = np.asarray((shape[0], shape[1] + shape_i[1]))
        # add to result
        datalist = np.concatenate((datalist, datalist_i))
        row_inds = np.concatenate((row_inds, row_inds_i))
        col_inds = np.concatenate((col_inds, col_inds_i))


    return datalist, row_inds, col_inds, shape
if __name__ == '__main__':
    acts = np.arange(10)

    steps = mk_possible_steps(acts, stepsz=2)

    print('num_of_steps: {} \n{}'.format(len(steps), steps))

    assert len(steps) == len(acts) ** 2

    steps1 = mk_possible_steps(acts, stepsz=3)

    print('num_of_steps: {} \n{}'.format(len(steps1), steps1))

    assert len(steps1) == len(acts) ** 3 
