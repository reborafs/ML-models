from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib
import tensorflow.compat.v2.feature_column as fc
import tensorflow as tf


# Importing Datasets.
datas_train = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
datas_eval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')
label_train = datas_train.pop('survived')
label_eval = datas_eval.pop('survived')

# Divide into categorical and numerical columns.
#    You can check the columns with datas_train.columns
CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck',
                       'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']


feature_columns = 