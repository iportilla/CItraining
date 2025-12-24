#!/usr/bin/env python

"""
NN_Model_1H.py

Python script for model tuning experiments.
Running this script requires four arguments on the command line:

    python3  NN_Model_1H.py  HIDDEN_NEURONS  LEARNING_RATE  BATCH_SIZE  EPOCHS

"""

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

import time
import json


# Utilize command line arguments to set the hyperparameters
HIDDEN_NEURONS = int(sys.argv[1])
LEARNING_RATE = float(sys.argv[2])
BATCH_SIZE = int(sys.argv[3])
EPOCHS = int(sys.argv[4])

# Create model output directory
MODEL_DIR = "model_1H" + str(HIDDEN_NEURONS) + "N_lr" + str(LEARNING_RATE) + "_bs" + str(BATCH_SIZE) + "_e" + str(EPOCHS)

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# print hyperparameter information
print()
print("Hyperparameters for the training:")
print("  - hidden_neurons:", HIDDEN_NEURONS)
print("  - learning_rate: ", LEARNING_RATE)
print("  - batch_size:    ", BATCH_SIZE)
print("  - epochs:        ", EPOCHS)
print()


# # 2. Loading Sherlock Applications Data

# Load in the pre-processed SherLock data and then split it into train and validation datasets
datafile = "sherlock/sherlock_18apps.csv"
df_orig, df, labels, df_labels_onehot, df_features = load_prep_data_18apps(datafile,print_summary=True)
train_features, val_features, train_labels, val_labels, train_L_onehot, val_L_onehot = split_data_18apps(df_features, labels, df_labels_onehot)


# Though we verified the loading and splitting of the pre-processed data, the following cells are just additional verification.

print("First 10 rows/entries from the preprocessed data:")
print(df.head(10))
print()

print("Last 10 rows/entries from the preprocessed data:")
print(df.tail(10))
print()

# ## 3. The NN model
model_1H = NN_Model_1H(HIDDEN_NEURONS,LEARNING_RATE)
model_1H_history = model_1H.fit(train_features,
                                train_L_onehot,
                                epochs=EPOCHS, batch_size=BATCH_SIZE,
                                validation_data=(val_features, val_L_onehot),
                                verbose=2)
# ## 4. Save the Output

# Utilize os.path.join to add the output files to the MODEL_DIR defined above.
history_file = os.path.join(MODEL_DIR, 'model_history.csv')
plot_file = os.path.join(MODEL_DIR, 'loss_acc_plot.png')
model_file = os.path.join(MODEL_DIR, 'model_weights.h5')
metadata_file = os.path.join(MODEL_DIR, 'model_metadata.json')

history_df = pd.DataFrame(model_1H_history.history)
history_df.to_csv(history_file, index=False)

combine_loss_acc_plots(model_1H_history,
              plot_loss, plot_acc,
              loss_epoch_shifts=(0, 1),
              acc_epoch_shifts=(0, 1), show=False,
              savefig_file=plot_file)


model_1H.save(model_file)

# Because of the terseness of Keras API, we create our own definition
# of a model metadata.

# timestamp of the results (at the time of saving)
model_1H_timestamp = time.strftime('%Y-%m-%dT%H:%M:%S%z')
# last epoch results is a key-value pair (i.e. a Series)
last_epoch_results = history_df.iloc[-1]

model_1H_metadata = {
    # Our own information
    'dataset': 'sherlock_18apps',
    'keras_version': tf.keras.__version__,
    'SLURM_JOB_ID': os.environ.get('SLURM_JOB_ID', None),
    'timestamp': model_1H_timestamp,
    'model_code': MODEL_DIR,
    'optimizer': 'Adam',
    # the number of hidden layers will be deduced from the length
    # of the hidden_neurons array:
    'hidden_neurons': [HIDDEN_NEURONS],
    'learning_rate': LEARNING_RATE,
    'batch_size': BATCH_SIZE,
    'epochs': EPOCHS,
    # Some results
    'last_results': {
        'loss': round(last_epoch_results['loss'], 8),
        'accuracy': round(last_epoch_results['accuracy'], 8),
        'val_loss': round(last_epoch_results['val_loss'], 8),
        'val_accuracy': round(last_epoch_results['val_accuracy'], 8),
    }
}

with open(metadata_file, 'w') as F:
    json.dump(model_1H_metadata, F, indent=2)