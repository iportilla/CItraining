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

from keras.models import Sequential, Model, model_from_json, load_model
from keras import regularizers
from keras.layers  import Dense, Dropout, Activation, Lambda, Flatten
from keras.layers import Input, ELU, LSTM, Embedding, Convolution2D, MaxPooling2D, \
   BatchNormalization, Convolution1D, MaxPooling1D, concatenate
from keras.preprocessing import sequence
from keras.optimizers import SGD, Adam, RMSprop
#from keras.utils import np_utils


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


def print_layers_dims(model):
    """Prints layer dimensions for any model.
    (GENERAL PURPOSE TOOL)
    """
    l_layers = model.layers
    # Note None is ALWAYS batch_size
    for i in range(len(l_layers)):
        print(l_layers[i])
        print('Input Shape: ', l_layers[i].input_shape, 'Output Shape: ', l_layers[i].output_shape)


def prepare_data(data_home, file_name, test_ratio=.2):
    # For simplicity we will save the prepared data to global vars
    global df_drones
    global all_FM, all_L, label_L
    global train_FM, train_L, dev_FM, dev_L

    df_drones = pandas.read_csv(os.path.join(data_home, file_name))

    df_features = df_drones.copy()
    del df_features["class"]
    # Unsplit feature matrix (all_FM) and label vector (all_L)
    all_FM = df_features.astype("float64").values
    all_L, label_L = categorical_to_numerics(df_drones["class"])
    train_FM, dev_FM, train_L, dev_L = train_test_split(all_FM, all_L,
                                                        test_size=test_ratio)

    return train_FM, train_L, dev_FM, dev_L   # notice shuffing of output results


def train_model(model, training_FM, training_L):
    fitted_model = model.fit(training_FM, training_L)
    return fitted_model

    
def model_predict(model, FM):
    return model.predict(FM)


def Model01_dense_no_hidden(inp_len=6):
    """Definition of deep learning model #1: no hidden layer
    """
    # Create an input layer
    main_input = Input(shape=(inp_len,), name='main_input')
    # The dense layer: This is also an output layer (last fully connected layer)
    # on this model.
    output = Dense(1, activation='sigmoid', name='output')(main_input)

    # Compile model and define optimizer
    model = Model(inputs=main_input, outputs=output)
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    return model


def Model02_dense_1(inp_len=6, hidden_len=64, W_reg=None):
    """Definition of deep Learning model #2: one hidden layer
    """
    # Create an input layer
    main_input = Input(shape=(inp_len,), name='main_input')
    # Hidden layer #1, taking input from `main_input`
    hidden1 = Dense(hidden_len, activation='relu')(main_input)
    # The dense layer: This is also an output layer (last fully connected layer) on this model.
    output = Dense(1, activation='sigmoid', name='output')(hidden1)

    # Compile model and define optimizer
    model = Model(inputs=main_input, outputs=output)
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    return model


def Model03_dense_2(inp_len=6, hidden_len=64, W_reg=None):
    """Definition of deep Learning model #3: two hidden layer
    """
    # Create an input layer
    main_input = Input(shape=(inp_len,), name='main_input')
    # Hidden layer #1, taking input from `main_input`
    hidden1 = Dense(hidden_len, activation='relu')(main_input)
    # Hidden layer #2
    hidden2 = Dense(hidden_len, activation='relu')(hidden1)
    # The dense layer: This is also an output layer (last fully connected layer) on this model.
    output = Dense(1, activation='sigmoid', name='output')(hidden2)

    # Compile model and define optimizer
    model = Model(inputs=main_input, outputs=output)
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    return model


def optimize_model(model, epochs=5, batch_size=32):
    """Runs an optimization ('training') on a neural network model.
    This must be done only after step01, step02, step03 were completed."""
    global train_FM, train_L, dev_FM, dev_L

    print("Optimizing KERAS model: ", model.__class__.__name__)
    print("Evaluation metric names: ", model.metrics_names)
    model_fit_hist = model.fit(train_FM, train_L,
                               epochs=epochs, batch_size=batch_size)

    print("Cross-validating...")
    loss, accuracy = model.evaluate(dev_FM, dev_L, verbose=1)

    print()
    print("Final Cross-Validation Accuracy", accuracy)
    print("Final Cross-Validation Loss    ", loss)
    print()
    print("Model summary:")
    model.summary()
    print_layers_dims(model)
    return model, model_fit_hist, loss, accuracy


def repeat_model(rep=10):
    for i in range(0,rep):
        model_nn = Model01_dense_no_hidden()
        print("\nRun ", i)
        #r = random()
        tr_fm, tr_l, dev_fm, dev_l = prepare_data(DATA_HOME, "machinelearningdata.csv", ratio = r)
        model, model_fit_hist, loss, accuracy = optimize_model(model_nn)


if __name__ == "__main__" and "get_ipython" not in globals():
    repeat_model()
