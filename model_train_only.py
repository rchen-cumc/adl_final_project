# model_training_only.py

import tensorflow as tf
print(tf.__version__)
assert tf.__version__.startswith('2')

from tensorflow.keras import Model
import matplotlib.pyplot as plt
from tensorflow.keras import layers

import pandas as pd
import numpy as np
import copy
import pickle
from tqdm import tqdm
from datetime import datetime
from dateutil.relativedelta import relativedelta
from preprocess_fn import *
from common_functions import *

# Hyperparameters
PAST_SIZE_CONDS = 125
PAST_SIZE_PROCS = 100
PAST_SIZE_DRUGS = 25
BATCH_SIZE = 256
EPOCHS = 15
embedding_dim = 128
class_weight = {0: 1.,
                1: 25.}

with open('/scratch/conds_dict.pickle', 'rb') as handle:
  conds_dict = pickle.load(handle)

with open('/scratch/conds_lookups.pickle', 'rb') as handle:
  conds_lookups = pickle.load(handle)

with open('/scratch/procs_dict.pickle', 'rb') as handle:
  procs_dict = pickle.load(handle)

with open('/scratch/procs_lookups.pickle', 'rb') as handle:
  procs_lookups = pickle.load(handle)

with open('/scratch/drugs_dict.pickle', 'rb') as handle:
  drugs_dict = pickle.load(handle)

with open('/scratch/drugs_lookups.pickle', 'rb') as handle:
  drugs_lookups = pickle.load(handle)

pt_labels = np.load('/scratch/Y.npy')
age_days = np.load('/scratch/age_days.npy')

PAST_SIZE_CONDS = 125
PAST_SIZE_PROCS = 100
PAST_SIZE_DRUGS = 25
BATCH_SIZE = 256
EPOCHS = 15
embedding_dim = 128
class_weight = {0: 1.,
                1: 25.}

conds_dense = convertConceptsToInds(conds_lookups['person_id_to_ind'], conds_lookups['concept_id_to_ind'], conds_dict)
procs_dense = convertConceptsToInds(procs_lookups['person_id_to_ind'], procs_lookups['concept_id_to_ind'], procs_dict)
drugs_dense = convertConceptsToInds(drugs_lookups['person_id_to_ind'], drugs_lookups['concept_id_to_ind'], drugs_dict)

# this will get changed.
# Set FILL_VAL to be the ind+1 of the dict
FILL_VAL_CONDS = len(conds_lookups['concept_id_to_ind']) + 1
FILL_VAL_PROCS = len(procs_lookups['concept_id_to_ind']) + 1
FILL_VAL_DRUGS = len(drugs_lookups['concept_id_to_ind']) + 1

conds_lookups['ind_to_concept_id'][FILL_VAL_CONDS] = 'MASK'
conds_lookups['concept_id_to_ind']['MASK'] = FILL_VAL_CONDS

procs_lookups['ind_to_concept_id'][FILL_VAL_PROCS] = 'MASK'
procs_lookups['concept_id_to_ind']['MASK'] = FILL_VAL_PROCS

drugs_lookups['ind_to_concept_id'][FILL_VAL_DRUGS] = 'MASK'
drugs_lookups['concept_id_to_ind']['MASK'] = FILL_VAL_DRUGS

# used to be list_of_lists
padded_data_conds = tf.keras.preprocessing.sequence.pad_sequences(
    conds_dense,
    maxlen=PAST_SIZE_CONDS,
    dtype='int32',
    padding='post',
    truncating='pre',
    value= FILL_VAL_CONDS
)

padded_data_procs = tf.keras.preprocessing.sequence.pad_sequences(
    procs_dense,
    maxlen=PAST_SIZE_PROCS,
    dtype='int32',
    padding='post',
    truncating='pre',
    value= FILL_VAL_PROCS
)

padded_data_drugs = tf.keras.preprocessing.sequence.pad_sequences(
    drugs_dense,
    maxlen=PAST_SIZE_DRUGS,
    dtype='int32',
    padding='post',
    truncating='pre',
    value= FILL_VAL_DRUGS
)

train_dataset = tf.data.Dataset.from_tensor_slices((padded_data_conds,
    padded_data_procs, padded_data_drugs, age_days, pt_labels))
train_dataset = train_dataset.batch(BATCH_SIZE)

# leaky ReLU activation
# lrelu = lambda x: tf.keras.layers.LeakyReLU(alpha=0.1)(x)

# conds branch
conds_input = tf.keras.layers.Input(name='condition_list', shape=(PAST_SIZE_CONDS,))
embed_conds = tf.keras.layers.Embedding(FILL_VAL_CONDS + 1, embedding_dim,
    input_length=PAST_SIZE_CONDS)(conds_input)
conds_LSTM  = tf.keras.layers.LSTM(32, activation= 'relu')(embed_conds)

# procs branch
procs_input = tf.keras.layers.Input(name='procedures_list', shape=(PAST_SIZE_PROCS,))
embed_procs = tf.keras.layers.Embedding(FILL_VAL_PROCS + 1, embedding_dim,
    input_length=PAST_SIZE_PROCS)(procs_input)
procs_LSTM  = tf.keras.layers.LSTM(32, activation= 'relu')(embed_procs)

# drugs branch
drugs_input = tf.keras.layers.Input(name='drugs_list', shape=(PAST_SIZE_DRUGS,))
embed_drugs = tf.keras.layers.Embedding(FILL_VAL_DRUGS + 1, embedding_dim,
    input_length=PAST_SIZE_DRUGS)(drugs_input)
drugs_LSTM  = tf.keras.layers.LSTM(32, activation= 'relu')(embed_drugs)

# structured age_days
age_days_input = tf.keras.layers.Input(name='age_in_days', shape=(1,))

# Merge branches
merged = tf.keras.layers.concatenate([conds_LSTM, procs_LSTM, drugs_LSTM, age_days_input])

# Connect with two dense layers
fc1 = tf.keras.layers.Dense(64, activation = 'relu')(merged)
output = tf.keras.layers.Dense(1, activation = 'sigmoid')(fc1)

model = Model(inputs=[conds_input, procs_input, drugs_input, age_days_input], outputs=output)

model.summary()

model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['binary_accuracy'])

for epoch in range(EPOCHS):
    print('EPOCH: ' + str(epoch + 1) + '/' + str(EPOCHS))
    for i, (conds_data, procs_data, drugs_data, age_day, pt_label) in tqdm(enumerate(train_dataset)):
        history = model.train_on_batch(x=(conds_data, procs_data, drugs_data, age_day), y=pt_label, class_weight=class_weight)
        print(history)

model.save('/model/my_model.h5')
