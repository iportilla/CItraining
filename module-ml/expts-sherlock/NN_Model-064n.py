#!/usr/bin/env python
"""
Complete Python script to do the ML pipeline of Sherlock 19F17C dataset
using two methods:

* Decision tree classifier
* Logistic regression

Use all remaining features.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
import sklearn
from sklearn import preprocessing
import tensorflow as tf

from analysis_sherlock_ML import *

# Import KERAS objects
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers

# CUSTOMIZATIONS
numpy.set_printoptions(linewidth=1000)


def model_evaluate(model,test_F,test_L):
    test_L_pred = model.predict(test_F)
    print("Evaluation by using model:",type(model).__name__)
    print("accuracy_score:",accuracy_score(test_L, test_L_pred))
    print("confusion_matrix:","\n",confusion_matrix(test_L, test_L_pred))
    return

# Notice the path:
df = pd.read_csv("../sherlock/sherlock_18apps.csv")
summarize_dataset(df)
df2 = preprocess_sherlock_19F17C(df)

print()
Rec = step0_label_features(df2)

print("After label-feature separation:")
print(Rec.df_features)
print(Rec.df_features.info())
tmp = step_onehot_encoding(Rec)
tmp = step_feature_scaling(Rec)
print("After scaling:")
print(Rec.df_features.head(10))

print()
tmp = step_train_test_split(Rec, test_size=0.2, random_state=34)

print()

# Neural network part is here

Rec.train_L_onehot = pd.get_dummies(Rec.train_labels)
Rec.test_L_onehot = pd.get_dummies(Rec.test_labels)

def NN_Model(hidden_neurons,learning_rate):
    """Definition of deep learning model with one dense hidden layer"""
    model = Sequential([
        Dense(hidden_neurons, activation='relu',input_shape=(19,),kernel_initializer='random_normal'),
        Dense(18, activation='softmax')
    ])
    adam=tf.keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    return model

print("Running neural network model now...")
model_1 = NN_Model(64,0.0003)
model_1.fit(Rec.train_features,
            Rec.train_L_onehot,
            epochs=10, batch_size=32,
            validation_data=(Rec.test_features, Rec.test_L_onehot),
            verbose=2)

#model_evaluate(model_1, Rec.test_features, Rec.test_L_onehot)
