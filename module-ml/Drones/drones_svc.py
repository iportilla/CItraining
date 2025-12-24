#!/usr/bin/env python
import os
import sys
import pandas
import numpy 
from numpy import random as rd
#import thundersvmScikit
#from thundersvmScikit import SVC
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

DATA_HOME = "/scratch-lustre/DeapSECURE/module03/Drones/data/"

def categorical_to_numerics(a, cats=None):
    """Converts array or Series of categorical/string values
    to an array of numeric enumeration.
    The input array `a` must be one-dimensional in nature.
    Unless the input categories are given (in `cats` argument),
    the labels will be auto-detected by this algorithm.

    Sometimes this step is needed because not all ML algorithms can
    take categoricals and go with it.
    """
    if cats is not None:
        # assume that cats is a valid list of categories
        pass
        # Otherwise, extract the categories: hopefully one of these
        # ways gets it:
    elif isinstance(a, pandas.Series):
        if isinstance(a.dtypes, pandas.api.types.CategoricalDtype):
            cats = a.dtypes.categories
        else:
            # general approach for array of strings
            cats = sorted(a.unique())
    elif isinstance(a, pandas.Categorical):
        cats = a.categories
    else:
        # general iterable case
        cats = sorted(pandas.Series(a).unique())

    # mapping: category -> numerics
    cat_map = dict((c, i) for (i,c) in enumerate(cats))
    # mapping: numerics -> category
    cat_revmap = list(cats)

    return (numpy.array([cat_map[c] for c in a]), cat_revmap)


def prepare_data(data_home, file_name, ratio=.8):
    data = pandas.read_csv(os.path.join(data_home, file_name))

    train_data, dev_data = train_test_split(data, test_size = ratio)
    train_labels = train_data["class"]
    train_fm = train_data.copy()
    del train_fm["class"]
    train_fm = train_fm.astype("float64").values
    train_real_labels, labels = categorical_to_numerics(train_labels)
    dev_fm = dev_data.copy()
    del dev_fm["class"]
    dev_fm = dev_fm.astype("float64").values
    dev_labels = dev_data["class"]
    dev_real_labels, dev_l_cat = categorical_to_numerics(dev_labels)

    return train_fm, train_real_labels, dev_fm, dev_real_labels


def train_model(model, training_data, training_labels):
    fitted_model = model.fit(training_data, training_labels)
    return fitted_model

    
def model_predict(model, testing_fm):
    return model.predict(testing_fm)
    

def repeat_model(rep=10):
    for i in range(0,rep):
        model_svc = SVC(verbose=1)
        r = rd.random()
        print("Run ", i, "  ratio is ",r)
        #r = random()
        tr_fm, tr_l, te_fm, te_l = prepare_data(DATA_HOME, "machinelearning_stdev.csv", ratio = r)
        tr_model = train_model(model_svc, tr_fm, tr_l)
        prediction = model_predict(tr_model, te_fm)

        conf_mat = confusion_matrix(te_l, prediction)
        accuracy = accuracy_score(te_l, prediction)

        print(conf_mat)
        print(accuracy)

# Only run the main program if calling this script from a regular python
# as a program:

if __name__ == "__main__" and "get_ipython" not in globals():
    repeat_model()
