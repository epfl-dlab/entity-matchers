
# Need to invert this:

"""
    for sent in [sentA, sentB]:
        for token in sent.split(' '):
            if token in ['COL', 'VAL']:
                res += token + ' '
            elif token in topk_tokens_copy:
                res += token + ' '
                topk_tokens_copy.remove(token)

        res += '\t'

    res += label + '\n'
    return res
"""
import os

import pandas as pd
import argparse

parser = argparse.ArgumentParser(
    description="Convert from ditto to deepamtcher"
)

parser.add_argument(
    "--input_dataset_folder",
    type=str,
    help=''
)

parser.add_argument(
    "--output_dataset_folder",
    type=str,
    help=''
)

args_main = parser.parse_args()

for fold in range(1, 6):
    try:
        os.makedirs("{}/721_5folds/{}".format(args_main.output_dataset_folder, fold))
    except FileExistsError:
        print("Fold {} already exists, override".format(fold))

    for in_file, out_file in zip(['train.txt.su.balance', 'test.txt.su', 'valid.txt.su'], ['train.csv', 'test.csv', 'valid.csv']):
        with open("{}/721_5folds/{}/{}".format(args_main.input_dataset_folder, fold, in_file)) as f:
            count_errors = 0
            obj_row = {"label": [], "id": [], "ltable_id": [], "ltable_names": [], "ltable_one_hop_names": [],
                       "ltable_one_hop_other_attributes": [],
                       "ltable_other_attributes": [], "rtable_id": [], "rtable_names": [], "rtable_one_hop_names": [],
                       "rtable_one_hop_other_attributes": [],
                       "rtable_other_attributes": [], 'ltable_relations': [], 'rtable_relations': []}
            for j, row in enumerate(f):

                errors = False
                sentA, sentB, label = row.strip().split('\t')
                objs = []
                for i, sent in enumerate([sentA, sentB]):
                    if not errors:
                        obj = {}
                        col_str = ""
                        val_str = ""
                        for token in sent.split(' '):
                            if token in ['COL', 'VAL']:
                                if token == "COL":
                                    col_name = True
                                    val_name = False
                                    if col_str != "":
                                        obj[col_str.strip(" ")] = val_str.strip(" ")
                                    col_str = ""
                                    val_str = ""
                                else:
                                    val_name = True
                                    col_name = False
                            else:
                                if col_name:
                                    col_str += token + " "
                                else:
                                    val_str += token + " "
                        for k in obj:
                            if (("ltable_" if i == 0 else "rtable_") + k) not in obj_row:
                                errors = True
                                count_errors += 1
                                print(obj)
                        objs.append(obj)
                if not errors:
                    keys = ['id', 'names', 'other_attributes', 'one_hop_names', 'one_hop_other_attributes', 'relations']
                    keys_covered = set()
                    for i, obj in enumerate(objs):
                        for k in obj:
                            obj_row[("ltable_" if i == 0 else "rtable_") + k].append(obj[k])
                    for k in keys:
                        for i in range(2):
                            if k not in objs[i]:
                                obj_row[("ltable_" if i == 0 else "rtable_") + k].append("")
                    obj_row['id'].append(j)
                    obj_row['label'].append(label)

            print("Errors: ", count_errors)
            df = pd.DataFrame(obj_row)
            df.to_csv("{}/721_5folds/{}/{}".format(args_main.output_dataset_folder, fold, out_file), index=False)

