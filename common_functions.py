import pandas as pd
import numpy as np
import copy
import pickle
from tqdm import tqdm
from tqdm import tqdm_notebook
from datetime import datetime
from dateutil.relativedelta import relativedelta
from collections import OrderedDict

# Get the useful rows for a given filename w/o .csv
def queryUsefulRows(filename):
    '''
    Return the useful rows given the OMOP_usefule_columns.csv has already been loaded into pandas.
    Input:
        filename - str, name of input file (e.g. 'person' or 'death'
    Output:
        useful_rows - list, ['col1', 'col2']
    '''
    useful_cols = pd.read_csv('OMOP_usefule_columns.csv')
    return list(useful_cols[(useful_cols['TabNam'] == filename) & (useful_cols['Useful'] == True)].ColNam)

# Create a corresponding per-patient lookup dictionary of events from a given OMOP CSV
# For each person_id (abbreviated pID), find all valid:
#     condition_occurrence [condition_concept_id, condition_start_date] # no end_date
#     drug_exposure        [drug_concept_id, drug_exposure_start_date]  # no end_date
#     measurement          [measurement_id, measurement_date] # want value?
#     observation          [observation_concept_id, observation_date] # observation_concept_id or observation_type_concept_id, want value?
#     observation_period   [period_type_concept_id, observation_period_end_date] # leave this out
#     procedure_occurrence [procedure_concept_id, procedure_date]  # vs procedure_type_concept_id
#     visit_occurrence     [visit_concept_id, visit_end_date] # visit_type_concept_id vs visit_concept_id
# that had an occurence. We parse for end date within 1 year of the last visit end date (LVED) in a separate function.
def queryTimelinesFromCSV(fp, column_names):
    '''
        For a given OMOP data CSV from fp, create a lookup table indexed by patient ID that contains
        column_names occurrences.
        Inputs:
            fp - filepath to OMOP CSV
            column_names - a two-element list of column_names to query; element 0 being the concept_id/element 1 being the time.
        Outputs:
            pID_concepts_dict - dictionary where key = person ID, value = zipped list of (concept_id, concept_date)
    '''

    # Split fp by '/' first, then split by period
    filename = fp.split('/')[-1].split('.')[0]

    # Let's get just the indices of the useful rows
    useful_row = queryUsefulRows(filename)

    # let's chunk with 100000 at a time
    # wc -l 34507185
    data_chunker = pd.read_csv(fp, chunksize=100000, usecols=useful_row)

    # person_id to list of tuples, [(condition_concept_id_1, condition_start_date_1),
    #                              (condition_concept_id_2, condition_start_date_2) ...]
    pID_concepts_dict = OrderedDict()

    for chunk in tqdm(data_chunker):
        # Sanity - drop duplicates
        chunk.drop_duplicates(inplace=True)
        chunk.dropna(axis=0, subset=[column_names[1]], inplace=True)

        # Get set of all person_ids in this chunk
        person_ids_chunk = set(chunk.person_id)

        # For each person_id, get one list of (for example) condition_concept_ids,
        # another list of (for example) condition_start_date
        # Look up datafield from input_arg
        # Zip together, and append to the dict
        for person_id in person_ids_chunk:
            # Chunk of dataframe for this person_id
            tmp = chunk[chunk['person_id'] == person_id]
            # Get one list of concept_ids
            concept_id_list = tmp[column_names[0]].tolist()
            # Get one list of dates
            # try:
            date_list = [datetime.strptime(x, '%Y-%m-%d') for x in tmp[column_names[1]].tolist()]
            # except TypeError:
            #    print(tmp[column_names[1]].tolist())
            #    sys.exit(1)

            # Zip together two lists into a list of tuples,
            save_list = list(zip(concept_id_list, date_list))
            # Save into dict
            if person_id in pID_concepts_dict:
                pID_concepts_dict[person_id] = pID_concepts_dict[person_id] + save_list;
            else:
                pID_concepts_dict[person_id] = save_list

            # persons.loc[(persons['person_id'] == person_id),'last_visit_end_date'] = last_visit_end_date

    return pID_concepts_dict

# Count how many possible inds there are, create lookup dicts
def queryAllConcepts(fp, column_name):
    '''
        For a given OMOP data CSV from fp, create a lookup table indexed by patient ID that contains
        column_names occurrences.
        Inputs:
            fp - filepath to OMOP CSV
            column_name - variable to query dataframe by
        Outputs:
            all_concept_ids - dictionary where key = person ID, value = zipped list of (concept_id, concept_date)
            concept_id_to_ind_dict -
            ind_to_concept_id_
    '''
    # Split fp by '/' first, then split by period
    # filename = fp.split('/')[-1].split('.')[0]

    all_concept_ids = set()
    validConcepts = pd.read_csv(fp, usecols=[column_name])
    validConcepts.dropna(axis=0, subset=[column_name], inplace=True)
    all_concept_ids = set(validConcepts[column_name].tolist())

    # Create a lookup for condition_concept_id to ind of numpy array (these are the columns)
    concept_id_to_ind_dict = OrderedDict()
    ind_to_concept_id_dict = OrderedDict()
    for ind, concept_id in enumerate(all_concept_ids):
        concept_id_to_ind_dict[concept_id] = ind
        ind_to_concept_id_dict[ind] = concept_id

    return all_concept_ids, concept_id_to_ind_dict, ind_to_concept_id_dict

def createPersonToInd(train=True):
    if train:
        persons = pd.read_csv('/scratch/persons_LVED.csv')
    else:
        persons = pd.read_csv('/scratch/persons_LVED_test.csv')

    # We also need a lookup for all person_ids (skip indices potentially)
    person_id_to_ind_dict = OrderedDict()
    ind_to_person_id_dict = OrderedDict()
    for ind, person_id in enumerate(set(persons.person_id.tolist())):
        person_id_to_ind_dict[person_id] = ind
        ind_to_person_id_dict[ind] = person_id

    return person_id_to_ind_dict, ind_to_person_id_dict

def removeEventsWithCutoff(pID_concepts_dict, pID_LVED_dict, concept_id_to_ind_dict, cutoff=6):
    # pID_concepts_dict = copy.deepcopy(in_pID_concepts_dict)
    new_dict = OrderedDict()
    for person_id in tqdm(pID_concepts_dict):
        # Last visit end date - 6 months
        # Something happened with typecasting...
        try:
            lved_cutoff = pID_LVED_dict[person_id] - relativedelta(months=+cutoff)
        except KeyError:
            # person never visited the hospital. delete key.
            # del pID_concepts_dict[person_id]
            continue;

        new_list = []

        for ele in list(pID_concepts_dict[person_id]):
            # If the date of the event is later than 6 months prior to the last_visit_end_date,
            # and occurs in >= 100 patients, keep. Otherwise, continue (pass)
            if ele[1] >= lved_cutoff and ele[0] in concept_id_to_ind_dict:
                new_list.append(ele)
            else:
                continue

        # bad practice, we're modifying a dictionary inside the function
        new_dict[person_id] = new_list

    return new_dict;
