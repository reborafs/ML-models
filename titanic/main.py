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


# Create Feature columns.
feature_columns = []

for feature_name in CATEGORICAL_COLUMNS:
    vocabulary =  datas_train[feature_name].unique()
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

# Creating input functions.
def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
    def input_function():
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
        if shuffle:
            ds = ds.shuffle(1000)
        ds = ds.batch(batch_size).repeat(num_epochs)
        return ds
    return input_function

train_input_fn = make_input_fn(datas_train, label_train)
eval_input_fn = make_input_fn(datas_eval, label_eval, num_epochs=1, shuffle=False)

# Creating the model.
linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)
linear_est.train(train_input_fn)
result = linear_est.evaluate(eval_input_fn)
clear_output()
accuracy = result['accuracy']