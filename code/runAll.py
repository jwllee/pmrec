#!/usr/bin/env python

import os, sys, argparse, time, subprocess
import pandas as pd
import numpy as np


if __name__ == '__main__':
    print('Python version: {}'.format(sys.version))
    # do the completed events
    print('Running experiment using run.py..')
    subprocess.call('python3 -W ignore ./PMRec/run.py', shell=True)

    # running experiment considering all events
    print('Running experiment using runLifecycle...')
    subprocess.call('python3 -W ignore ./PMRec/runLifecycle.py', shell=True)

    # consider all events and with a different log matrix representation
    print('Running experiment using all events and a diff. log matrix...')
    subprocess.call('python3 -W ignore ./PMRec/runLifecycleActivity.py', shell=True)
    
