import pandas as pd
import numpy as np
import os, sys
from fastFM.datasets import make_user_item_regression
from sklearn.cross_validation import train_test_split
from fastFM import als


if __name__=='__main__':
	print('Python version: {}'.format(sys.version))
	X, y, _ = make_user_item_regression(label_stdev=.4)
	X_train, X_test, y_train, y_test = train_test_split(X, y)
	
	fm = als.FMRegression(n_iter=1000, init_stdev=0.1, rank=2, l2_reg_w=0.1, l2_reg_V=0.5)
	fm.fit(X_train, y_train)
	y_pred = fm.predict(X_test)
	print('Predictions:\n{}'.format(y_pred))
	
	print(d)
