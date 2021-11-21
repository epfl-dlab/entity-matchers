import py_entitymatching as em
import os
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(
    description="Convert dataset to ditto format"
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


def convert_table_ab_to_ditto_file(df, output_file):
    rows = []
    columns = ['id', 'names', 'other_attributes', 'one_hop_names', 'one_hop_other_attributes', 'relations']
    for i, l in df.iterrows():
        string_row = ""
        for c in columns:
            string_row += "COL {} ".format(c)
            string_row += "VAL {} ".format(str(l[c]).replace("nan", ""))
        string_row += "\n"
        rows.append(string_row)
    with open(output_file, 'w') as f:
        for l in rows:
            f.write(l)


if __name__ == "__main__":
    """
    format is 
    <entry_1> \t <entry_2> \t <label>
    
    where label is 0 or 1
    entry_1 is 
    COL title VAL microsoft visio standard 2007 version upgrade COL manufacturer VAL microsoft COL price VAL 129.95
    etc
    """
    try:
        os.makedirs("{}/721_5folds/".format(args_main.output_dataset_folder))
    except FileExistsError:
        pass
    for fold in range(1, 6):
        print("Fold {}".format(fold))
        for file in ['train.csv', 'test.csv', 'valid.csv', 'test_cross_product.csv']:
            print("File {}".format(file))
            if file not in os.listdir("{}/721_5folds/{}".format(args_main.input_dataset_folder, fold)):
                print("File {} could not be found in {}/721_5folds/{}".format(file, args_main.input_dataset_folder, fold))
            else:
                try:
                    os.makedirs("{}/721_5folds/{}".format(args_main.output_dataset_folder, fold))
                except FileExistsError:
                    pass
                df_input = pd.read_csv("{}/721_5folds/{}/{}".format(args_main.input_dataset_folder, fold, file))
                columns_left = sorted([col for col in df_input.columns if 'ltable_' in col])
                columns_right = sorted([col for col in df_input.columns if 'rtable_' in col])
                out_strings = []
                for i, l in df_input.iterrows():
                    string_row = ""
                    for c in columns_left:
                        string_row += "COL {} ".format(c.replace("ltable_", ""))
                        string_row += "VAL {} ".format(str(l[c]).replace("nan", ""))
                    string_row += "\t"
                    for c in columns_right:
                        string_row += "COL {} ".format(c.replace("rtable_", ""))
                        string_row += "VAL {} ".format(str(l[c]).replace("nan", ""))
                    string_row += "\t" + str(l['label']) + "\n"
                    out_strings.append(string_row)
                with open("{}/721_5folds/{}/{}".format(args_main.output_dataset_folder, fold, file.replace(".csv", ".txt")), 'w') as f:
                    for l in out_strings:
                        f.write(l)

    # Convert tableA, tableB
    df_a = pd.read_csv("{}/tableA.csv".format(args_main.input_dataset_folder))
    df_b = pd.read_csv('{}/tableB.csv'.format(args_main.input_dataset_folder))

    convert_table_ab_to_ditto_file(df_a, "{}/table_a.txt".format(args_main.output_dataset_folder))
    convert_table_ab_to_ditto_file(df_b, "{}/table_b.txt".format(args_main.output_dataset_folder))

