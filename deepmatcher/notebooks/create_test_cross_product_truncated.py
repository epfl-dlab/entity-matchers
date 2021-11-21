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
    description="Given a dataset split in different folds (using create_dataset_deepmatchers), create the test set with the cross product truncated to 50 neighbours",
)
parser.add_argument(
    "--dataset",
    type=str,
    help='must contain the folder 721_5folds'
)
parser.add_argument(
    "--dataset_deepmatcher",
    type=str,
    help='must contain the files tableA.csv, tableB.csv and 721_5folds'
)
parser.add_argument(
    "--name_coeff",
    type=float,
    help='from 1, 1 is important, > 1 is less important.'
)
parser.add_argument(
    "--other_attribute_coeff",
    type=float,
    help='from 1, 1 is important, > 1 is less important.'
)
parser.add_argument(
    "--one_hop_name_coeff",
    type=float,
    help='from 1, 1 is important, > 1 is less important.'
)
parser.add_argument(
    "--one_hop_other_attribute_coeff",
    type=float,
    help='from 1, 1 is important, > 1 is less important.'
)
parser.add_argument(
    "--neighbours",
    type=float,
    help='50 or 25, number of neighbours per entity in the test set.'
)
parser.add_argument(
    "--out_filename",
    type=str,
    help='name of the output filename.'
)
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
    df_a = pd.read_csv('{}/tableA.csv'.format(dataset_deepmatcher))
    df_b = pd.read_csv('{}/tableB.csv'.format(dataset_deepmatcher))
    return train_folds, test_folds, valid_folds, df_a, df_b

# global variables, a bit bad!
train_folds, test_folds, valid_folds, df_a, df_b = get_data(args_main.dataset, args_main.dataset_deepmatcher)

df_a.columns = ["ltable_" + x for x in df_a.columns]
df_b.columns = ["rtable_" + x for x in df_b.columns]

print("len of df_a and df_b: {}, {}".format(len(df_a), len(df_b)))


def fatina(input_args):
    new_test_links = []
    e1 = input_args[0]
    e2 = input_args[1]

    e1_row = df_a.loc[e1]
    e2_row = df_b.loc[e2]
    jac_sim_e1_to_b = {}
    jac_sim_e2_to_a = {}
    for n, oa, oh_n, oh_oa, id_ent in zip(df_b['rtable_names'], df_b['rtable_other_attributes'],
                                      df_b['rtable_one_hop_names'], df_b['rtable_one_hop_other_attributes'],
                                      df_b.index):
        if len(str(e1_row['ltable_names'])) > 3 and len(str(n)) > 3:
            jac_sim_e1_to_b[id_ent] = jaccard.distance(e1_row['ltable_names'], n) * args_main.name_coeff
        elif len(str(e1_row['ltable_other_attributes'])) > 3 and len(str(oa)) > 3:
            jac_sim_e1_to_b[id_ent] = jaccard.distance(e1_row['ltable_other_attributes'],
                                                   oa) * args_main.other_attribute_coeff
        elif len(str(e1_row['ltable_one_hop_names'])) > 3 and len(str(oh_n)) > 3:
            jac_sim_e1_to_b[id_ent] = jaccard.distance(e1_row['ltable_one_hop_names'],
                                                   oh_n) * args_main.one_hop_name_coeff
        elif len(str(e1_row['ltable_one_hop_other_attributes'])) > 3 and len(str(oh_oa)) > 3:
            jac_sim_e1_to_b[id_ent] = jaccard.distance(e1_row['ltable_one_hop_other_attributes'],
                                                   oh_oa) * args_main.one_hop_other_attribute_coeff
    for n, oa, oh_n, oh_oa, id_ent in zip(df_a['ltable_names'], df_a['ltable_other_attributes'],
                                      df_a['ltable_one_hop_names'], df_a['ltable_one_hop_other_attributes'],
                                      df_a.index):
        if len(str(e2_row['rtable_names'])) > 3 and len(str(n)) > 3:
            jac_sim_e2_to_a[id_ent] = jaccard.distance(e2_row['rtable_names'], n) * args_main.name_coeff
        elif len(str(e2_row['rtable_other_attributes'])) > 3 and len(str(oa)) > 3:
            jac_sim_e2_to_a[id_ent] = jaccard.distance(e2_row['rtable_other_attributes'],
                                                   oa) * args_main.other_attribute_coeff
        elif len(str(e2_row['rtable_one_hop_names'])) > 3 and len(str(oh_n)) > 3:
            jac_sim_e2_to_a[id_ent] = jaccard.distance(e2_row['rtable_one_hop_names'],
                                                   oh_n) * args_main.one_hop_name_coeff
        elif len(str(e2_row['rtable_one_hop_other_attributes'])) > 3 and len(str(oh_oa)) > 3:
            jac_sim_e2_to_a[id_ent] = jaccard.distance(e2_row['rtable_one_hop_other_attributes'],
                                                   oh_oa) * args_main.one_hop_other_attribute_coeff
    jac_sim_e1_to_b_sorted = dict(sorted(jac_sim_e1_to_b.items(), key=lambda item: item[1]))
    jac_sim_e2_to_a_sorted = dict(sorted(jac_sim_e2_to_a.items(), key=lambda item: item[1]))

    added_1 = 0
    added_2 = 0
    sims = []
    for k in jac_sim_e1_to_b_sorted:
        if (e1, k) not in train_folds and (e1, k) not in valid_folds:
            new_test_links.append({"ltable_id": e1, "rtable_id": k})
            added_1 += 1
            sims.append(jac_sim_e1_to_b_sorted[k])
        if added_1 == args_main.neighbours:
            break
    for k in jac_sim_e2_to_a_sorted:
        if (k, e2) not in train_folds and (k, e2) not in valid_folds:
            new_test_links.append({"ltable_id": k, "rtable_id": e2})
            added_2 += 1
            sims.append(jac_sim_e2_to_a_sorted[k])
        if added_2 == args_main.neighbours:
            break
    assert len(new_test_links) <= args_main.neighbours * 2
    return new_test_links


if __name__ == "__main__":
    jaccard = Jaccard(3)
    # For the 5 folds, create test sets so that the first 50 neighbours are considered, without considering train and valid in the folds.
    df_a.set_index('ltable_id', inplace=True)
    df_b.set_index('rtable_id', inplace=True)
    for fold in range(5):
        with multiprocessing.Pool(20) as p:
            results = list(tqdm(p.imap(fatina, [(e1, e2) for e1, e2 in list(test_folds[fold])])))

        pairs_not_correct = [len(x) != args_main.neighbours * 2 for x in results]
        print("Pairs not correct (they don't add 100 rows) are: {}".format(np.sum(pairs_not_correct)))
        results_flatten = [l for x in results for l in x]

        df_new_pairs = pd.DataFrame(results_flatten)
        df_new_pairs.drop_duplicates(inplace=True)
        start = time.time()
        df_new_pairs.set_index('ltable_id', inplace=True)

        df_test_merged = df_new_pairs.merge(df_a, left_index=True, right_index=True)

        df_test_merged.reset_index(inplace=True)

        df_test_merged.set_index('rtable_id', inplace=True)

        df_test_merged = df_test_merged.merge(
            df_b, left_on='rtable_id', right_on='rtable_id')
        df_test_merged.reset_index(inplace=True)
        print("time elapsed {}".format(time.time() - start))
        df_test_merged['id'] = list(range(len(df_test_merged)))
        start = time.time()
        labels = df_test_merged.apply(lambda x: 1 if (x['ltable_id'], x['rtable_id']) in test_folds[fold] else 0, axis=1)

        assert len(labels) == len(df_new_pairs)
        print("time elapsed for labels {}".format(time.time() - start))
        df_test_merged['label'] = labels
        print("Number of true positives: {}".format(np.sum(labels)))
        print("Length of dataset: {}".format(len(df_test_merged)))
        start = time.time()
        df_test_merged.to_csv(
            "{}/721_5folds/{}/{}".format(args_main.dataset_deepmatcher, fold+1, args_main.out_filename),
            index=False)
        print("time elapsed for print {}".format(time.time() - start))


