# model_infer_only.py

#################
#### INFER ######
#################

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
from tqdm import tqdm_notebook
from datetime import datetime
from dateutil.relativedelta import relativedelta
from preprocess_fn import *
from collections import OrderedDict

PAST_SIZE_CONDS = 125
PAST_SIZE_PROCS = 100
PAST_SIZE_DRUGS = 25

with open('/scratch/conds_dict_test.pickle', 'rb') as handle:
  conds_dict = pickle.load(handle)

with open('/scratch/conds_lookups_test.pickle', 'rb') as handle:
  conds_lookups = pickle.load(handle)

with open('/scratch/procs_dict_test.pickle', 'rb') as handle:
  procs_dict = pickle.load(handle)

with open('/scratch/procs_lookups_test.pickle', 'rb') as handle:
  procs_lookups = pickle.load(handle)

with open('/scratch/drugs_dict_test.pickle', 'rb') as handle:
  drugs_dict = pickle.load(handle)

with open('/scratch/drugs_lookups_test.pickle', 'rb') as handle:
  drugs_lookups = pickle.load(handle)

age_days = np.load('/scratch/age_days_test.npy')

# conds_mat_test = createMatrix(conds_lookups['person_id_to_ind'], conds_lookups['concept_id_to_ind'], conds_dict)
assert conds_lookups['person_id_to_ind'] == procs_lookups['person_id_to_ind']
assert conds_lookups['person_id_to_ind'] == drugs_lookups['person_id_to_ind']

conds_dense = convertConceptsToInds(conds_lookups['person_id_to_ind'], conds_lookups['concept_id_to_ind'], conds_dict)
procs_dense = convertConceptsToInds(procs_lookups['person_id_to_ind'], procs_lookups['concept_id_to_ind'], procs_dict)
drugs_dense = convertConceptsToInds(drugs_lookups['person_id_to_ind'], drugs_lookups['concept_id_to_ind'], drugs_dict)

model = tf.keras.models.load_model('/model/my_model.h5')

#cond_dataset_test = tf.data.Dataset.from_tensor_slices((conds_mat_test))

model.summary()

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

predictions=model.predict((padded_data_conds, padded_data_procs, padded_data_drugs, age_days.astype(np.float32)))

match_output = OrderedDict()
for i, pID in enumerate(conds_lookups['person_id_to_ind']):
    match_output[pID] = predictions[i]

persons = pd.read_csv('/infer/person.csv', usecols=['person_id'])

final_out = []
for i in range(len(persons)):
    final_out.append(match_output[persons.iloc[i].person_id][0])

persons['score'] = final_out
persons.to_csv('/output/predictions.csv')
