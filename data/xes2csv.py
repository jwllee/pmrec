import pandas as pd
import numpy as np
import os, sys
from pandas.api.types import CategoricalDtype

from pm4py.objects.log.importer.xes import factory as xes_importer


if __name__ == '__main__':
    fp = 'BPI_Challenge_2012.xes.gz'
    log = xes_importer.import_log(fp)

    caseid_list = []
    event_list = []
    trans_list = []

    # -1 to include all cases
    # n_cases = -1
    n_cases = 100
    i = 0

    for trace in log:
        caseid = trace.attributes['concept:name']
        events = map(lambda e: e['concept:name'], trace)
        events = list(events)
        trans = map(lambda e: e['lifecycle:transition'], trace)
        trans = list(trans)

        caseids = [caseid for _ in range(len(events))]

        caseid_list.extend(caseids)
        event_list.extend(events)
        trans_list.extend(trans)

        i += 1
        if n_cases > 0 and i >= n_cases:
            break

    log_df = pd.DataFrame({
        'caseId': caseid_list,
        'activity': event_list,
        'lifecycle': trans_list
    })

    ordered_acts = sorted(log_df['activity'].unique())
    act_cat = CategoricalDtype(ordered_acts, ordered=True)
    log_df['activity_id'] = log_df['activity'].astype(act_cat).cat.codes + 1

    print(log_df.head())
    if n_cases < 0:
        out_fp = 'BPIC2012.csv'
    else:
        out_fp = 'BPIC2012-len_{}.csv'.format(n_cases)
    log_df.to_csv(out_fp, index=None)
