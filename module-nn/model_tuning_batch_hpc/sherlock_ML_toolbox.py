"""Data preparation & common tools for sherlock_18apps dataset.

This library is part of the DeapSECURE / T3-CIDERS projects.
Root repo: https://gitlab.com/deapsecure/deapsecure-mod03-ml-devel/
See LICENSE.md at the root repo for copyright and licensing
of this software.

This library is supposed to be constructed by learners bit-by-bit.
The library will start with the data preparation for ML
(see the code in `Prep_ML.py`)
that has been packaged in two user-callable functions,
and the necessary values are returned in tuples.
This complete toolkit is given to learners to speed up their hands-on activities;
but the rest of the codes that would not likely be covered
in most workshops should be readable by new practitioners of Python.


SUGGESTION FOR USAGE
--------------------

For better control, I would suggest that functions are imported
one by one, e.g.

    from sherlock_ML_toolbox import load_prep_data_18apps, \
                                    split_data_18apps ...

instead of

    from sherlock_ML_toolbox import *

to avoid gotchas with unexpected symbols.


DESIGN GUIDING PRINCIPLES
-------------------------

### Principle 1: minimal import

Try to strongly LIMIT imports here to only the standard modules already
imported explicitly at the start of this module.
All the other function/class names should be imported
in the individual functions to minimize namespace flooding.
In particular, AVOID importing neural network libraries
here because they would cause this library excessively heavy
for traditional ML work.


### Principle 2: use simple Python constructs

Try to minimize classes (as much as possible),
convoluted logic, unclear / dubious variable names.
Try to minimize deeply nested calls (e.g. function calling function
calling function calling ...) because these would be hard to track down.
"""

import os
import sys

import pandas as pd
import numpy as np
import sklearn

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
#from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

import matplotlib.pyplot as plt

# CUSTOMIZATIONS (optional)
np.set_printoptions(linewidth=1000)


def load_prep_data_18apps(datafile,
                          print_summary=True):
    """Loads the sherlock_18apps dataset and do the preprocessing for
    ML and NN modeling: data cleaning, label/feature separation,
    feature normalization/scaling, etc. until it is ready for ML
    except for train-validation splitting.

    Args:
      datafile: The pathname of the data file. This must be a CSV file.
        The conventional file name is "sherlock/sherlock_18apps.csv".
      print_summary (optional): A switch to print info about the dataset or
        some summary or excerpt of the data.

    Returns:
      A 5-element tuple containing the following:

      (
        df,                  # (DataFrame) the original raw data
        df2,                 # (DataFrame) the cleaned data
        labels,              # (Series) the cleaned labels, original format
        df_labels_onehot,    # (DataFrame) the cleaned labels, one-hot encoded
        df_features          # (DataFrame) the cleaned features, preprocessed for ML
      )
    """
    print("Loading input data from: %s" % (datafile,))
    df = pd.read_csv(datafile, index_col=0)

    ## Summarize the dataset
    if print_summary:
        print("* shape:", df.shape)
        print()
        print("* info::\n")
        df.info()
        print()
        print("* describe::\n")
        print(df.describe().T)
        print()

    """Perform cleaning of a Sherlock 19F17C dataset.
    All the obviously bad and missing data are removed.
    """
    # Missing data or bad data
    del_features_bad = [
        'cminflt', # all-missing feature
        'guest_time', # all-flat feature
    ]
    df2 = df.drop(del_features_bad, axis=1)

    print("Cleaning:")
    print("- dropped", len(del_features_bad), "columns: ", del_features_bad)

    print("- remaining missing data (per feature):")
    isna_counts = df2.isna().sum()
    print(isna_counts[isna_counts > 0])

    print("- dropping the rest of missing data")
    df2.dropna(inplace=True)

    print("- remaining shape: %s" % (df2.shape,))
    print()

    """Separate labels from the features"""
    print("Step: Separating the labels (ApplicationName) from the features.")
    labels = df2['ApplicationName']
    df_features = df2.drop('ApplicationName', axis=1)

    """One-hot encoding: labels"""
    df_labels_onehot = pd.get_dummies(labels)

    """Perform one-hot encoding for **all** categorical features."""
    print("Step: Converting all non-numerical features to one-hot encoding.")
    df_features = pd.get_dummies(df_features)

    """Step: Feature scaling using StandardScaler."""
    print("Step: Feature scaling with StandardScaler")

    df_features_unscaled = df_features
    scaler = preprocessing.StandardScaler()
    scaler.fit(df_features_unscaled)

    # Recast the features still in a dataframe form
    df_features = pd.DataFrame(scaler.transform(df_features_unscaled),
                               columns=df_features_unscaled.columns,
                               index=df_features_unscaled.index)
    if print_summary:
        print("After scaling:")
        print(df_features.head(10))
    print()

    return \
        df, \
        df2, \
        labels, \
        df_labels_onehot, \
        df_features

    

def split_data_18apps(df_features,
                      labels,
                      df_labels_onehot,
                      val_size=0.2,
                      split_randseed=34):
    """Performs data splitting (train-test split) into training and
    validation datasets.

    Args:
      df_features (DataFrame): The (cleaned & scaled) feature matrix.
      labels (Series): The cleaned labels, original format.
      df_labels_onehot (DataFrame): The cleaned labels, one-hot encoded.
      val_size (float in the range (0, 1.0)): Fraction of data reserved
        for validation.
      split_randseed (int > 0): Random seed number used for the train_test_split
        procedure.

    Returns:
      A 6-element tuple containing the following:

      (
        train_features,      # (DataFrame) training set, feature matrix
        val_features,        # (DataFrame) validation set, feature matrix
        train_labels,        # (Series) training set, labels in original format
        val_labels,          # (Series) validation set, labels in original format
        train_L_onehot,      # (DataFrame) training set, labels in one-hot encoding
        val_L_onehot         # (DataFrame) validation set, labels in one-hot encoding
      )
    """

    """Step: Perform train-test split on the master dataset.
    This should be the last step before constructing & training the model.
    """
    print("Step: Train-test split  val_size=%s  random_state=%s" \
          % (val_size, split_randseed))

    train_features, val_features, train_labels, val_labels = \
        train_test_split(df_features, labels,
                         test_size=val_size, random_state=split_randseed)

    print("- training dataset: %d records" % (len(train_features),))
    print("- testing dataset:  %d records" % (len(val_features),))
    sys.stdout.flush()

    # Post-split the one-hot reps of the labels (classes) here,
    # which are needed for neural networks modeling.
    train_L_onehot = df_labels_onehot.loc[train_labels.index]
    val_L_onehot = df_labels_onehot.loc[val_labels.index]

    print("Now the dataset is ready for machine learning!")

    return \
        train_features, val_features, \
        train_labels, val_labels, \
        train_L_onehot, val_L_onehot


def NN_Model_1H(hidden_neurons, learning_rate, metrics=['accuracy']):
    """Defines and compiles a deep learning model with one dense hidden layer.

    Args:
      hidden_neurons (int): The number of neurons in the first (hidden) Dense layer.
      learning rate (float > 0): The learning rate for the Adam optimizer.
      metrics (list of metrics): Metrics to be computed and reported during model training.

    Returns:
      The Sequential NN model created.
    """
    # tools for deep learning:
    import tensorflow as tf
    import tensorflow.keras as keras

    # Import key Keras objects
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.optimizers import Adam

    # Use TensorFlow random normal initializer
    random_normal_init = tf.random_normal_initializer(mean=0.0, stddev=0.05)
    model = Sequential([
        # More hidden layers can be added here
        Dense(hidden_neurons, activation='relu', input_shape=(19,),
              kernel_initializer=random_normal_init), # Hidden Layer
        Dense(18, activation='softmax',
              kernel_initializer=random_normal_init)  # Output Layer
    ])
    adam_opt = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(optimizer=adam_opt,
                  loss='categorical_crossentropy',
                  metrics=metrics)

    print("Created model: NN_Model_1H")
    print(" - hidden_layers  = 1")
    print(" - hidden_neurons = {}".format(hidden_neurons))
    print(" - optimizer      = Adam")
    print(" - learning_rate  = {}".format(learning_rate))
    print()

    return model



def plot_loss(model_history, epoch_shifts=None, show=True,
              fmt_train='-o', label_train='Train Loss',
              fmt_val='-x', label_val='Val Loss'):
    """Plot the progression of loss function during an NN model training,
    given the model's history. Loss computed with both the training and
    validation datasets during the training process are plotted in one graph.

    Args:
      model_history (History): The History object returned from NN model training
        function, Model.fit() .
      epoch_shifts (2-tuple of ints): An optional epoch shift to align the x-axis
        of the plot. The first element represents the epoch shift for the training
        loss, and the second one for the validation dataset's loss.
      show (bool): Whether to show the plot after created.

    Returns:
      The current Figure object.
    """
    # Set default if not given a value for epoch_shifts.
    if epoch_shifts is None:
        epoch_shifts = (0, 1)
    # Set the x-axis (epochs).
    epochs_train = np.array(model_history.epoch) + epoch_shifts[0]
    epochs_val = np.array(model_history.epoch) + epoch_shifts[1]

    # Plot the training epochs vs. training loss.
    plt.plot(epochs_train,
             model_history.history['loss'],
             fmt_train,
             label=label_train)
    # Plot the validation epochs vs. validation loss.
    plt.plot(epochs_val,
             model_history.history['val_loss'],
             fmt_val,
             label=label_val)
    # Set the title as well as the x and y-axes.
    plt.title('Model Loss', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.xlabel('Epoch', fontsize=14)
    # Set x-axis limits based on the known epoch values.
    plt.xlim([min(np.min(epochs_train), np.min(epochs_val)),
              max(np.max(epochs_train), np.max(epochs_val))])
    # Place legend in the upper right corner.
    plt.legend(loc='upper right')
    # Adjust x-axis and y-axis font size.
    plt.tick_params(axis='x', labelsize=14)
    plt.tick_params(axis='y', labelsize=14)
    fig = plt.gcf()

    if show:
        plt.show()

    # Return the current figure if further manipulations are needed
    return fig


def plot_acc(model_history, epoch_shifts=None, show=True):
    """Plot the progression of accuracy during an NN model training,
    given the model's history. Accuracy computed with the training and
    validation datasets during the training process are plotted in one graph.

    Args:
      model_history (History): The History object returned from NN model training
        function, Model.fit() .
      epoch_shifts (2-tuple of ints): An optional epoch shift to align the x-axis
        of the plot. The first element represents the epoch shift for the training
        loss, and the second one for the validation dataset's loss.
      show (bool): Whether to show the plot after created.

    Returns:
      The current Figure object.
    """
    # Set default if not given a value for epoch_shifts.
    if epoch_shifts is None:
        epoch_shifts = (0, 1)
    # Set the x-axis (epochs).
    epochs_train = np.array(model_history.epoch) + epoch_shifts[0]
    epochs_val = np.array(model_history.epoch) + epoch_shifts[1]
    # Plot the training epochs vs. training accuracy.
    plt.plot(epochs_train,
             model_history.history['accuracy'],
             '-o',
             label='Train Accuracy')
    # Plot the validation epochs vs. validation accuracy.
    plt.plot(epochs_val,
             model_history.history['val_accuracy'],
             '-x',
             label='Val Accuracy')
    # Set the title as well as the x and y-axes.
    plt.title('Model Accuracy', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.xlabel('Epoch', fontsize=14)
    # Set x-axis limits based on the known epoch values.
    plt.xlim([min(np.min(epochs_train), np.min(epochs_val)),
              max(np.max(epochs_train), np.max(epochs_val))])
    # Place legend in the lower right corner.
    plt.legend(loc='lower right')
    # Adjust x-axis and y-axis font size.
    plt.tick_params(axis='x', labelsize=14)
    plt.tick_params(axis='y', labelsize=14)
    fig = plt.gcf()

    if show:
        plt.show()

    # Return the current figure if further manipulations are needed
    return fig


def combine_loss_acc_plots(model_history,
                           plot_loss_func=plot_loss,
                           plot_acc_func=plot_acc,
                           figsize=(10.0, 5.0),
                           loss_epoch_shifts=None,
                           acc_epoch_shifts=None,
                           show=True,
                           wspace=0.4,
                           savefig_file=None):
    """Creates a combined, side-by-side loss and accuracy plots
    using a given model's history.
    This function allows adjustment on many parameters such as
    the plot size, epoch shifts, etc.

    Args:
      model_history (History): The History object from the NN model training
        function, Model.fit().
      plot_loss_func (function): The function that plots the loss function.
        Defaults to this module's `plot_loss` function.
      plot_acc_func (function): The function that plots the accuracy.
        Defaults to this module's `plot_acc` function.
      figsize (tuple): The size of the figure in 2-tuple (width, height).
        Experimentation may be needed to find desirable figure dimensions.
      loss_epoch_shifts (2-tuple of ints): An optional epoch shifts to align
        the x-axis of the loss plot: see notes below.
      acc_epoch_shifts (2-tuple of ints): An optional epoch shifts to align
        the x-axis of the accuracy plot: see notes below.
      show (bool): Whether to show the plot after created.
      wspace (float): The width (horizontal) spacing reserved between subplots,
        expressed as a fraction of the average axis width.
      figsave_file (string or file-like object): If defined, specifies the target
        object (file) to which the figure will be saved.

    Returns:
      The current Figure object.


    This function demonstrates the possibility of compositing plots
    which are generated using other functions into a single plot panel.


    Notes on epoch shifts:
    ----------------------
    The two epoch shift parameters (loss_epoch_shift and acc_epoch_shifts)
    are intended to adjust the epoch number displayed on the horizontal axes
    of the plots. Each of them takes the form of (2-tuple of ints),
    where the first (second) int represents the epoch shift for the loss or accuracy
    computed with the training (validation) data.
    Under normal circumstances, the epoch shifts should be identical for the loss
    and accuracy plots.
    """

    # Set the figure's size.
    plt.figure(figsize=figsize)
    # Specify the location of the first subplot.
    plt.subplot(1, 2, 1)
    # Call the function to plot the model's loss and give it the appropriate inputs.
    fig1 = plot_loss_func(model_history, loss_epoch_shifts, show=False)
    # Specify the location of the second subplot.
    plt.subplot(1, 2, 2)
    # Call the function to plot the model's accuracy and give it the appropriate inputs.
    fig2 = plot_acc_func(model_history, acc_epoch_shifts, show=False)
    # Set the horizontal spacing.
    plt.subplots_adjust(wspace=wspace)

    fig = plt.gcf()
    if savefig_file is not None:
        plt.savefig(savefig_file)

    if show:
        plt.show()

    # Return the current figure if further manipulations are needed
    return fig



##
## Standard File/Directory Naming Conventions
## (for neural network tuning experiments)
##

def fn_dir_tuning_1H(expt_name,
                     hidden_neurons, learning_rate,
                     batch_size, epochs):
    """
    Constructs a run/output directory name for a given NN model tuning
    experiment name, a set of hyperparameters
    (hidden neurons, learning rate, batch size),
    and num of epochs.

    This case supports only 1-hidden-layer case (hence, "1H"),
    so `hidden_neurons` must be a non-negative integer.

    Special note for learning rate, which is a float parameter:
    we adopt the following convention to produce the easiest-to-read
    value:

      - We will use a fixed-width decimal format up to 9 digits
        after the decimal point.
      - But we will strip excess 0's beyond what's necessary.
        For example, '0.002500000' will be stripped to become '0.0025' .
      - We will further strip '.' if there are no nonzero digits
        after the decimal point.
    """
    if isinstance(learning_rate, str):
        lr_str = learning_rate
    else:
        lr_str = f"{learning_rate:.9f}".rstrip("0").rstrip(".")
    expt_subdir = f"model_1H{hidden_neurons}N_lr{lr_str}_bs{batch_size}_e{epochs}"
    return os.path.join(expt_name, expt_subdir)


def fn_out_history_1H(expt_name,
                      hidden_neurons, learning_rate,
                      batch_size, epochs):
    """
    Constructs the history data filename for a given NN model tuning
    experiment (1 hidden layer) and a set of hyperparameters.
    Consult the documentation of `fn_dir_tuning_1H` for more details.
    """
    return os.path.join(fn_dir_tuning_1H(expt_name,
                                         hidden_neurons, learning_rate,
                                         batch_size, epochs),
                        'model_history.csv')


def fn_out_metadata_1H(expt_name,
                       hidden_neurons, learning_rate,
                       batch_size, epochs):
    """
    Constructs the metadata data filename for a given NN model tuning
    experiment (1 hidden layer) and a set of hyperparameters.
    Consult the documentation of `fn_dir_tuning_1H` for more details.
    """
    return os.path.join(fn_dir_tuning_1H(expt_name,
                                         hidden_neurons, learning_rate,
                                         batch_size, epochs),
                        'model_metadata.json')


def fn_out_model_1H(expt_name,
                    hidden_neurons, learning_rate,
                    batch_size, epochs):
    """
    Constructs the model checkpoint filename for a given NN model tuning
    experiment (1 hidden layer) and a set of hyperparameters.
    Consult the documentation of `fn_dir_tuning_1H` for more details.

    Despite the name (`model_weights.h5`), this file actually contains
    a complete model information, including the structure of the model
    and the weights and biases.
    """
    return os.path.join(fn_dir_tuning_1H(expt_name,
                                         hidden_neurons, learning_rate,
                                         batch_size, epochs),
                        'model_weights.h5')


def fn_out_plot_1H(expt_name,
                   hidden_neurons, learning_rate,
                   batch_size, epochs):
    """
    Constructs the loss&acc plot filename for a given NN model tuning
    experiment (1 hidden layer) and a set of hyperparameters.
    Consult the documentation of `fn_dir_tuning_1H` for more details.

    Despite the name (`model_weights.h5`), this file actually contains
    a complete model information, including the structure of the model
    and the weights and biases.
    """
    return os.path.join(fn_dir_tuning_1H(expt_name,
                                         hidden_neurons, learning_rate,
                                         batch_size, epochs),
                        'loss_acc_plot.png')


### Support for Multiple Hidden Layers


def model_layer_code_XH(hidden_neurons):
    """Constructs a model-layer code string (e.g. 1H18N, 2H32N18N, ...).
    """
    hidden_neurons = list(hidden_neurons)
    hn_str = str(len(hidden_neurons)) + "H" \
           + "".join(str(HN) + "N" for HN in hidden_neurons)
    return hn_str


def fn_dir_tuning_XH(expt_name,
                     hidden_neurons, learning_rate,
                     batch_size, epochs):
    """
    Constructs a run/output directory name for a given NN model tuning
    experiment name, a set of hyperparameters
    (hidden neurons, learning rate, batch size),
    and num of epochs.

    This case supports the variable-hidden-layer case (hence, "XH"),
    so `hidden_neurons` must be a list or tuple of non-negative integers.

    Special note for learning rate, which is a float parameter:
    we adopt the following convention to produce the easiest-to-read
    value:

      - We will use a fixed-width decimal format up to 9 digits
        after the decimal point.
      - But we will strip excess 0's beyond what's necessary.
        For example, '0.002500000' will be stripped to become '0.0025' .
      - We will further strip '.' if there are no nonzero digits
        after the decimal point.
    """
    hidden_neurons = list(hidden_neurons)
    hn_str = str(len(hidden_neurons)) + "H" \
           + "".join(str(HN) + "N" for HN in hidden_neurons)
    if isinstance(learning_rate, str):
        lr_str = learning_rate
    else:
        lr_str = f"{learning_rate:.9f}".rstrip("0").rstrip(".")
    expt_subdir = f"model_{hn_str}_lr{lr_str}_bs{batch_size}_e{epochs}"
    return os.path.join(expt_name, expt_subdir)


def fn_out_history_XH(expt_name,
                      hidden_neurons, learning_rate,
                      batch_size, epochs):
    """
    Constructs the history data filename for a given NN model tuning
    experiment (variable hidden layer) and a set of hyperparameters.
    Consult the documentation of `fn_dir_tuning_XH` for more details.
    """
    return os.path.join(fn_dir_tuning_XH(expt_name,
                                         hidden_neurons, learning_rate,
                                         batch_size, epochs),
                        'model_history.csv')



