#!/usr/bin/env python
# coding: utf-8

# # Results of converting a Jupyter Notebook to a Python Script.
# 
# This code can now be submitted to an HPC job scheduler.
# 
# This script contains the full code to train the "baseline model" in the model tuning process (see the ["Tuning Neural Network Models for Better Accuracy"](https://deapsecure.gitlab.io/deapsecure-lesson04-nn/30-model-tuning/) of [DeapSECURE module 4: Deep Learning (Neural Network)](https://deapsecure.gitlab.io/deapsecure-lesson04-nn/)).
# 
# The code in this notebook will train the neural network model to classify among 18 different mobile phone apps using the `sherlock_18apps` dataset. The baseline model has one hidden layer with 18 neurons, with learning rate 0.0003 and batch size 32.

# ## 1. Loading Python Libraries

import os
import sys

import pandas as pd
import numpy as np

# CUSTOMIZATIONS (optional)
np.set_printoptions(linewidth=1000)

# tools for deep learning:
import tensorflow as tf
import tensorflow.keras as keras

# Import ML toolbox functions
from sherlock_ML_toolbox import load_prep_data_18apps, split_data_18apps, \
NN_Model_1H, plot_loss, plot_acc, combine_loss_acc_plots

# # 2. Loading Sherlock Applications Data

# Load in the pre-processed SherLock data and then split it into train and validation datasets
datafile = "sherlock/sherlock_18apps.csv"

df_orig, df, labels, df_labels_onehot, df_features = load_prep_data_18apps(datafile, print_summary=True)

train_features, val_features, train_labels, val_labels, train_L_onehot, val_L_onehot = split_data_18apps(df_features, labels, df_labels_onehot)


# Though we verified the loading and splitting of the pre-processed data, the following cells are just additional verification.

print("First 10 rows/entries from the preprocessed data:")
print(df.head(10))
print()

print("Last 10 rows/entries from the preprocessed data:")
print(df.tail(10))
print()

print("Info:")
print(df_features.info())
print()

print("First 10 feature entries:")
print(df_features.head(10))
print()

print("Number of records in the datasets:")
print("- training dataset:", len(train_features), "records")
print("- valing dataset:", len(val_features), "records")
sys.stdout.flush()

print("Now the feature matrix is ready for machine learning!")

print("First 10 training feature entries:")
print(train_features.head(10))
print()

print("The number of entries for each application name:")
app_counts = df.groupby('ApplicationName')['CPU_USAGE'].count()
print(app_counts)
print("Num of applications:", len(app_counts))
print()

# ## 3. The Baseline Model
model_1H = NN_Model_1H(18,0.0003)
model_1H_history = model_1H.fit(train_features,
                                train_L_onehot,
                                epochs=10, batch_size=32,
                                validation_data=(val_features, val_L_onehot),
                                verbose=2)

# ## 4. Save the Output

history_file = 'model_1H18N_history.csv'
plot_file = 'loss_acc_plot.png'
model_file = 'model_1H18N.h5'

history_df = pd.DataFrame(model_1H_history.history)
history_df.to_csv(history_file, index=False)


combine_loss_acc_plots(model_1H_history,
              plot_loss, plot_acc,
              loss_epoch_shifts=(0, 1),
              acc_epoch_shifts=(0, 1), show=False,
              savefig_file=plot_file)

model_1H.save(model_file)