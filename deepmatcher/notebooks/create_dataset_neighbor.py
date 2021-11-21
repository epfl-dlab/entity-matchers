import multiprocessing
import time

import deepmatcher as dm
# Import py_entitymatching package
import py_entitymatching as em
import os
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
import argparse
from strsimpy.qgram import QGram
from strsimpy.jaccard import Jaccard


parser = argparse.ArgumentParser(
    description="Given a dataset split in different folds and tableA.csv and tableB.csv, "
                "create the dataset set with the cross product and top-k-neighbours, for train, valid and test.",
)
parser.add_argument(
    "--dataset_original",
    type=str,
    help='must contain the folder 721_5folds'
)
parser.add_argument(
    "--dataset_deepmatcher",
    type=str,
    help='must contain the files tableA.csv, tableB.csv, folds are not needed'
)
parser.add_argument(
    "--neighbors",
    type=int,
    help='50 or 25, number of neighbors per entity in the test set.'
)
parser.add_argument(
    "--out_dataset",
    type=str,
    help='Directory, should be something like DB-WD-15K-5NEIGH or something like that.'
)

NAME_COEFFICIENT = 1.0
OTHER_ATTRIBUTES_COEFFICIENT = 0.75
ONE_HOP_NAME_COEFFICIENT = 0.5
ONE_HOP_OTHER_ATTRIBUTES_COEFFICIENT = 0.25

args_main = parser.parse_args()


def get_data(dataset, dataset_deepmatcher):
    train_folds = []
    test_folds = []
    valid_folds = []
    for i in range(1, 6):
        train_folds.append(set())
        test_folds.append(set())
        valid_folds.append(set())
        for file, set_fold in zip(['train_links', 'test_links', 'valid_links'], [train_folds, test_folds, valid_folds]):
            with open("{}/721_5folds/{}/{}".format(dataset, i, file)) as f:
                for l in f:
                    e1, e2 = l.rstrip("\n").split('\t')
                    set_fold[-1].add((e1, e2))
    df_a = pd.read_csv('{}/tableA.csv'.format(dataset_deepmatcher))[['names', 'other_attributes', 'one_hop_names',
                                                                     'one_hop_other_attributes', 'relations', 'id']]
    df_b = pd.read_csv('{}/tableB.csv'.format(dataset_deepmatcher))[['names', 'other_attributes', 'one_hop_names',
                                                                     'one_hop_other_attributes', 'relations', 'id']]
    return train_folds, test_folds, valid_folds, df_a, df_b


# global variables, a bit bad!
train_folds, test_folds, valid_folds, df_a, df_b = get_data(args_main.dataset_original, args_main.dataset_deepmatcher)


def compute_jaccard_from_entity_to_df(e_df):
    """
    Taken one entity, and the other dataframe that needs to be matched, returns the top k by jaccard.
    Parameters
    ----------
    e: dataframe row for entity e
    df: other dataframe.

    Returns
    -------

    """
    e = e_df[0]
    df = e_df[1]
    jac_sim_e_to_df = {}  # Use similarity, the higher the better
    for n, oa, oh_n, oh_oa, id_ent in zip(df['names'], df['other_attributes'],
                                          df['one_hop_names'], df['one_hop_other_attributes'],
                                          df['id']):
        jac_sim_e_to_df[id_ent] = 0.
        if len(str(e['names'])) > 3 and len(str(n)) > 3:
            jac_sim_e_to_df[id_ent] += jaccard.similarity(e['names'], n) * NAME_COEFFICIENT
        elif len(str(e['other_attributes'])) > 3 and len(str(oa)) > 3:
            jac_sim_e_to_df[id_ent] += jaccard.similarity(e['other_attributes'], oa) * OTHER_ATTRIBUTES_COEFFICIENT
        elif len(str(e['one_hop_names'])) > 3 and len(str(oh_n)) > 3:
            jac_sim_e_to_df[id_ent] += jaccard.similarity(e['one_hop_names'], oh_n) * ONE_HOP_NAME_COEFFICIENT
        elif len(str(e['one_hop_other_attributes'])) > 3 and len(str(oh_oa)) > 3:
            jac_sim_e_to_df[id_ent] += jaccard.similarity(e['one_hop_other_attributes'], oh_oa) * ONE_HOP_OTHER_ATTRIBUTES_COEFFICIENT

    # Sorted by decreasing jaccard similarity. Most similar first. Return the first top k most similar entities
    return e['id'], list(dict(sorted(jac_sim_e_to_df.items(),
                                     key=lambda item: item[1],
                                     reverse=True)[:args_main.neighbors]).keys())  # keys are the entity IDs.


if __name__ == "__main__":
    jaccard = Jaccard(3)
    df_a.to_csv("{}/tableA.csv".format(args_main.out_dataset), index=False)
    df_b.to_csv("{}/tableB.csv".format(args_main.out_dataset), index=False)
    # For the 5 folds, create test sets so that the first 50 neighbours are considered, without considering train and valid in the folds.


    with multiprocessing.Pool(20) as p:
        results_1 = list(tqdm(p.imap(compute_jaccard_from_entity_to_df, [(e1, df_b) for i, e1 in df_a.iterrows()])))

    with multiprocessing.Pool(20) as p:
        results_2 = list(tqdm(p.imap(compute_jaccard_from_entity_to_df, [(e2, df_a) for i, e2 in df_b.iterrows()])))

    assert len(results_1) == len(df_a) and len(results_2) == len(df_b)
    assert np.all([len(r) == args_main.neighbors for e, r in results_1])
    assert np.all([len(r) == args_main.neighbors for e, r in results_2])

    df_a.columns = ['ltable_' + x for x in df_a.columns]
    df_b.columns = ['rtable_' + x for x in df_b.columns]

    results = set()  # Drop the duplicates!
    for e, neigh in results_1:
        for n in neigh:
            results.add((e, n))
    for e, neigh in results_2:
        for n in neigh:
            results.add((n, e))  # Reversed!!
    results = list(results)   # Make a list out of the set.

    for fold in range(1, 6):
        test_links = set()
        train_links = set()
        valid_links = set()
        files = ["train_links", 'valid_links', 'test_links']
        for file, list_ in zip(files, [train_links, valid_links, test_links], ):
            with open("{}/721_5folds/{}/{}".format(args_main.dataset_original, fold, file)) as f:
                for l in f:
                    e1, e2 = l.rstrip("\n").split("\t")
                    list_.add((e1, e2))

        test_positives = [(e1, e2) for (e1, e2) in results if (e1, e2) in test_links]
        valid_positives = [(e1, e2) for (e1, e2) in results if (e1, e2) in valid_links]  # All training positives (validation)
        train_positives = [(e1, e2) for (e1, e2) in train_links]   # All training positives.

        results_wrong = [(e1, e2) for (e1, e2) in results if ((e1, e2) not in test_links and (e1, e2) not in valid_links and (e1, e2) not in train_links)]

        print("{} should be >= {}".format(len(results_wrong) + len(test_positives) + len(valid_positives) + len(train_positives), len(results)))

        split_test = int( (len(test_positives) / (len(test_positives) + len(valid_positives) + len(train_positives))) * len(results_wrong))
        split_train = int((len(train_positives) / (len(test_positives) + len(valid_positives) + len(train_positives))) * len(results_wrong)) + split_test
        # split_valid = len(valid_positives) / (len(test_positives) + len(valid_positives) + len(train_positives))  this is useless

        df_test = pd.DataFrame([{"ltable_id": e1, "rtable_id": e2} for (e1, e2) in
                                test_positives + results_wrong[0: split_test]]).drop_duplicates()
        df_train = pd.DataFrame([{"ltable_id": e1, "rtable_id": e2} for (e1, e2) in
                                 train_positives + results_wrong[split_test: split_train]]).drop_duplicates()
        df_valid = pd.DataFrame([{"ltable_id": e1, "rtable_id": e2} for (e1, e2) in
                                 valid_positives + results_wrong[split_train: ]]).drop_duplicates()

        df_test = df_test.merge(df_a, left_on='ltable_id', right_on='ltable_id').merge(df_b, left_on='rtable_id', right_on='rtable_id')
        df_train = df_train.merge(df_a, left_on='ltable_id', right_on='ltable_id').merge(df_b, left_on='rtable_id',
                                                                                       right_on='rtable_id')
        df_valid = df_valid.merge(df_a, left_on='ltable_id', right_on='ltable_id').merge(df_b, left_on='rtable_id',
                                                                                       right_on='rtable_id')
        df_test['label'] = df_test[['ltable_id', 'rtable_id']].apply(lambda x:
                                                                     1 if (x['ltable_id'], x['rtable_id']) in test_links else 0,
                                                                     axis=1)
        df_test['id'] = [x for x in range(len(df_test))]
        df_train['label'] = df_train[['ltable_id', 'rtable_id']].apply(lambda x:
                                                                       1 if (x['ltable_id'], x['rtable_id']) in train_links else 0,
                                                                       axis=1)
        df_train['id'] = [x for x in range(len(df_train))]
        df_valid['label'] = df_valid[['ltable_id', 'rtable_id']].apply(lambda x:
                                                                       1 if (x['ltable_id'], x['rtable_id']) in valid_links else 0,
                                                                       axis=1)
        df_valid['id'] = [x for x in range(len(df_valid))]

        print("Fold {}".format(fold))
        print("TP in test: {}, len of test: {}, pos/neg ratio: {}".format(df_test['label'].sum(), len(df_test), df_test['label'].sum() / len(df_test)))
        print("TP in train: {}, len of train: {}, pos/neg ratio: {}".format(df_train['label'].sum(), len(df_train), df_train['label'].sum() / len(df_train)))
        print("TP in valid: {}, len of valid: {}, pos/neg ratio: {}".format(df_valid['label'].sum(), len(df_valid), df_valid['label'].sum() / len(df_valid)))

        if "721_5folds" not in os.listdir(args_main.out_dataset):
            os.makedirs(args_main.out_dataset + "/721_5folds")
        if str(fold) not in os.listdir("{}/721_5folds".format(args_main.out_dataset)):
            os.makedirs("{}/721_5folds/{}".format(args_main.out_dataset, fold))

        df_train.to_csv("{}/721_5folds/{}/train.csv".format(args_main.out_dataset, fold), index=False)
        df_test.to_csv("{}/721_5folds/{}/test.csv".format(args_main.out_dataset, fold), index=False)
        df_valid.to_csv("{}/721_5folds/{}/valid.csv".format(args_main.out_dataset, fold), index=False)

