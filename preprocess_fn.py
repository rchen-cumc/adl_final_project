import pandas as pd
import numpy as np
import copy
import pickle
from tqdm import tqdm
from tqdm import tqdm_notebook
from datetime import datetime
from dateutil.relativedelta import relativedelta
from common_functions import *
from collections import OrderedDict

def preprocess_LVED_deaths(train=True):
    #useful_cols = pd.read_csv('OMOP_usefule_columns.csv')
    if train:
        folder_fp = '/train/'
        death = pd.read_csv(folder_fp + 'death.csv', usecols = ['person_id',
            'death_date', 'death_type_concept_id'])
        fp = '/train/visit_occurrence.csv'
        save_append = ''
    else:
        folder_fp = '/infer/'
        fp = '/infer/visit_occurrence.csv'
        save_append = '_test'

    person_col_list= ['person_id','gender_concept_id',
    'year_of_birth','month_of_birth',
    'day_of_birth', 'birth_datetime',
    'race_concept_id', 'ethnicity_concept_id']

    persons = pd.read_csv(folder_fp + 'person.csv', usecols = person_col_list)

    # Only in train do we have this
    if train:
        death_ids = list(death['person_id'])
        persons['died'] = persons['person_id'].isin(death_ids)

    pID_LVED_dict = queryLastVisitEndDate(fp)
    person_id_list = persons['person_id'].tolist()

    # For the synpuf, I'm filling using 1970s as dummy
    fill_val = datetime(1970, 1, 1, 0, 0)
    LVED_list = []
    for person_id in person_id_list:
        try:
            LVED_list.append(pID_LVED_dict[person_id])
        except KeyError:
            LVED_list.append(fill_val)

    persons['last_visit_end_date'] = LVED_list

    if train:
        # Save in persons table a column of "label"
        # 0 - default if person did not die
        # 0 - if person died, but after 6 months from last visit
        # 1 - if person died between 0 and 6 months after last visit

        persons['label'] = 0;
        # Get all spots in dataframe where last_visit_end_date is not empty, and the person died
        for person_id in tqdm(persons[(~persons['last_visit_end_date'].isnull()) & (persons['died'])]['person_id']):
            # Lookup the death date from death
            death_date = datetime.strptime(death[death['person_id'] == person_id]['death_date'].values[0], '%Y-%m-%d')

            # Last visit end date + 6 months
            # Something happened with typecasting...
            lved_6m = pd.to_datetime(persons[persons.person_id == person_id]['last_visit_end_date'].values[0]) + relativedelta(months=+6)

            # Check if the death_date is less than or equal to the last visit + 6 months
            if death_date <= lved_6m:
                persons.loc[(persons['person_id'] == person_id),'label'] = 1

    # Let's also find their age at LVED (birth day - LVED) and add that as a column
    birthdays = []
    age = []
    for ind, row in persons.iterrows():
        # print(ind, row.person_id)
        # print(row.year_of_birth, row.month_of_birth, row.day_of_birth)
        try:
            birthday_str = str(int(row.year_of_birth)) + '-' + str(int(row.month_of_birth)) + '-' + str(int(row.day_of_birth))
            birthday = datetime.strptime(birthday_str, '%Y-%m-%d')
            birthdays.append(birthday)
            tdelta = row.last_visit_end_date - birthday
            age.append(tdelta.days)
        except ValueError:
            # nans? What to do...
            birthdays.append(fill_val)
            age.append(0)

    persons['birthday'] = birthdays
    persons['age'] = age

    age_days = persons['age'].to_numpy()
    np.save('/scratch/age_days' + save_append + '.npy', age_days)

    # Now, we save a copy of this persons_table, and the LVED dictionary.
    persons.to_csv('/scratch/persons_LVED' + save_append + '.csv', index=False)

    with open('/scratch/pID_LVED_dict' + save_append + '.pickle', 'wb') as handle:
        pickle.dump(pID_LVED_dict, handle, pickle.HIGHEST_PROTOCOL)

    if train:
        Y = persons['label'].to_numpy()
        np.save('/scratch/Y.npy', Y)

def queryLastVisitEndDate(fp):
    # Let's get just the indices of the useful rows
    useful_cols = ['visit_occurrence_id','person_id','visit_concept_id',
            'visit_start_date','visit_end_date','visit_type_concept_id']

    # let's chunk with 100000 at a time
    # wc -l 3353654
    visit_occurrence_chunker = pd.read_csv(fp, chunksize=100000, usecols=useful_cols)

    # person_id to last_visit_end_date dict
    pID_LVED_dict = OrderedDict()

    for chunk in tqdm(visit_occurrence_chunker):
	# Let's drop duplicates and NaNs
        chunk.drop_duplicates(inplace=True)
        chunk.dropna(axis=0, subset=['visit_end_date'], inplace=True)

        # Get set of all person_ids in this chunk
        person_ids_chunk = set(chunk.person_id)

        # For each person_id, get a list of last_visit_end_dates
        for person_id in person_ids_chunk:
            # Chunk of dataframe for this person_id
            tmp = chunk[chunk['person_id'] == person_id]
            # Get all visit_end_dates and find the max (last date)
            last_visit_end_date = max([datetime.strptime(x, '%Y-%m-%d') for x in tmp.visit_end_date.tolist()])
            # if key exists check if greater than existing last date, if so reset
            if person_id in pID_LVED_dict:
                if last_visit_end_date > pID_LVED_dict[person_id]:
                    pID_LVED_dict[person_id] = last_visit_end_date;
            else:
                pID_LVED_dict[person_id] = last_visit_end_date

            # persons.loc[(persons['person_id'] == person_id),'last_visit_end_date'] = last_visit_end_date

    return pID_LVED_dict

# Get the useful rows for a given filename w/o .csv
def queryUsefulRows(filename):
    '''
    Return the useful rows given the OMOP_usefule_columns.csv has already been loaded into pandas.
    Input:
        filename - str, name of input file (e.g. 'person' or 'death'
    Output:
        useful_rows - list, ['col1', 'col2']
    '''
    return list(useful_cols[(useful_cols['TabNam'] == filename) & (useful_cols['Useful'] == True)].ColNam)


# Functions for data processing, to make things neater
def createMatrix(person_id_to_ind_dict, concept_id_to_ind_dict, pID_concepts_dict):
    # Create numpy array of conditions for all patients.
    # e.g. x-axis: person_id, y_axis: 0-7765 (see ind_to_condition_concept_id_dict)
    data_mat = np.zeros((len(person_id_to_ind_dict), len(concept_id_to_ind_dict)))
    for person_id in tqdm(pID_concepts_dict):
        # [row = person_id, column = all concept_id_cols]
        data_mat[person_id_to_ind_dict[person_id],
                       [concept_id_to_ind_dict[x[0]] for x in pID_concepts_dict[person_id]]] += 1
    return data_mat

def make_cond_proc_drug(train=True):
    if train:
        folder_fp = '/train/'
        save_append = ''
    else:
        folder_fp = '/infer/'
        save_append = '_test'

    print('Folder for data loaded: ' + folder_fp)

    with open('/scratch/pID_LVED_dict' + save_append + '.pickle', 'rb') as handle:
        pID_LVED_dict = pickle.load(handle)

    conds_dict, conds_concept_ids, conds_to_ind_dict, ind_to_conds_dict, conds_person_id_to_ind_dict = run_data_pipeline(
        folder_fp + 'condition_occurrence.csv', ['condition_concept_id', 'condition_start_date'],
        '/app/condition_occurrence_concepts.csv', pID_LVED_dict, train)

    procs_dict, procs_concept_ids, procs_to_ind_dict, ind_to_procs_dict, procs_person_id_to_ind_dict = run_data_pipeline(
        folder_fp + 'procedure_occurrence.csv', ['procedure_concept_id', 'procedure_date'],
        '/app/procedure_occurrence_concepts.csv', pID_LVED_dict, train)

    drugs_dict, drugs_concept_ids, drugs_to_ind_dict, ind_to_drugs_dict, drugs_person_id_to_ind_dict = run_data_pipeline(
        folder_fp + 'drug_exposure.csv', ['drug_concept_id', 'drug_exposure_start_date'],
        '/app/drug_exposure_concepts.csv', pID_LVED_dict, train)

    with open('/scratch/conds_dict' + save_append + '.pickle', 'wb') as handle:
        pickle.dump(conds_dict, handle, pickle.HIGHEST_PROTOCOL)

    with open('/scratch/procs_dict' + save_append + '.pickle', 'wb') as handle:
        pickle.dump(procs_dict, handle, pickle.HIGHEST_PROTOCOL)

    with open('/scratch/drugs_dict' + save_append + '.pickle', 'wb') as handle:
        pickle.dump(drugs_dict, handle, pickle.HIGHEST_PROTOCOL)

    conds_lookups = {}
    conds_lookups['concept_ids'] = conds_concept_ids
    conds_lookups['concept_id_to_ind'] = conds_to_ind_dict
    conds_lookups['ind_to_concept_id'] = ind_to_conds_dict
    conds_lookups['person_id_to_ind'] = conds_person_id_to_ind_dict

    with open('/scratch/conds_lookups' + save_append + '.pickle', 'wb') as handle:
        pickle.dump(conds_lookups, handle, pickle.HIGHEST_PROTOCOL)

    procs_lookups = {}
    procs_lookups['concept_ids'] = procs_concept_ids
    procs_lookups['concept_id_to_ind'] = procs_to_ind_dict
    procs_lookups['ind_to_concept_id'] = ind_to_procs_dict
    procs_lookups['person_id_to_ind'] = procs_person_id_to_ind_dict

    with open('/scratch/procs_lookups' + save_append + '.pickle', 'wb') as handle:
        pickle.dump(procs_lookups, handle, pickle.HIGHEST_PROTOCOL)

    drugs_lookups = {}
    drugs_lookups['concept_ids'] = drugs_concept_ids
    drugs_lookups['concept_id_to_ind'] = drugs_to_ind_dict
    drugs_lookups['ind_to_concept_id'] = ind_to_drugs_dict
    drugs_lookups['person_id_to_ind'] = drugs_person_id_to_ind_dict

    with open('/scratch/drugs_lookups' + save_append + '.pickle', 'wb') as handle:
        pickle.dump(drugs_lookups, handle, pickle.HIGHEST_PROTOCOL)

def run_data_pipeline(fp, column_names, standard_lookup_file, pID_LVED_dict, train):
    print('Querying timelines...')
    pID_concepts_dict = queryTimelinesFromCSV(fp, column_names)
    print('Generating all concepts...')
    all_concept_ids, concept_id_to_ind_dict, ind_to_concept_id_dict = queryAllConcepts(
        standard_lookup_file, column_names[0])
    print('Creating person ID lookup')
    person_id_to_ind_dict, ind_to_person_id_dict = createPersonToInd(train)
    print('Using cutoffs to eliminate extraneous data...')
    final_pID_concepts_dict = removeEventsWithCutoff(pID_concepts_dict,
        pID_LVED_dict, concept_id_to_ind_dict, cutoff=6)
    # print('Creating numpy data matrix...')
    # data_mat = createMatrix(person_id_to_ind_dict, concept_id_to_ind_dict, final_pID_concepts_dict)
    return final_pID_concepts_dict, all_concept_ids, concept_id_to_ind_dict, ind_to_concept_id_dict, person_id_to_ind_dict

def convertConceptsToInds(person_id_to_ind_dict, concept_id_to_ind_dict, pID_concepts_dict):
    # Similar to createMatrix, but just directly do the conversion.
    # dense_mat is a list of lists:
    # dense_mat = [[0, 1, 4], [2, 5, 6], ...] where the internal
    # list contains the index lookup for the concept_id.
    dense_mat = []
    for pID in person_id_to_ind_dict:
        # if pID is not in conds_dict, then indData is an empty array.
        if pID not in pID_concepts_dict:
            dense_mat.append([])
        else:
            # Sort the conds_dict by second element in tuple (datetime)
            currData = sorted(pID_concepts_dict[pID], key = lambda x: x[1])
            # Convert from the full conceptIDs to their indices
            indData  = [concept_id_to_ind_dict[x[0]] for x in currData]
            dense_mat.append(indData)

    # Make sure dense_mat and person_id_to_ind are same input_length
    assert len(dense_mat) == len(person_id_to_ind_dict)
    return dense_mat
