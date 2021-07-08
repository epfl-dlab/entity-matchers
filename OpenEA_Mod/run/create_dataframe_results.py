import pandas as pd

import sys
import numpy as np
import argparse
import re

# Parse command line arguments (the names of the files to parse).
parser = argparse.ArgumentParser(
    description='Parse the log files provided as input and compute average and'
                ' standard deviation for the metrics hit1, hit5, '
                'hit10, hit50, mr, mrr among the files provided as input.')

parser.add_argument('-m', '--method',
                    help='name of the method either RDGCN or BootEA (careful with the spelling)',
                    required=True,
                    choices=['BootEA', 'RDGCN'])

parser.add_argument('-d', '--dataset',
                    help='name of the dataset, for example D_W_15K_V1',
                    required=True)

parser.add_argument('-f', '--files',
                    nargs='+',
                    help='list of input log files names',
                    required=True)

parser.add_argument('-form', '--output_format',
                    help='format for the output table (either xls either csv)',
                    required=True,
                    choices=['csv', 'xls'])

parser.add_argument('-o', '--output_file',
                    help='file for the output table (be sure to use the right extension, .xls or .csv)',
                    required=True)


args = parser.parse_args()


def create_df_data_entry(method: str, dataset: str, measure: str, results: list, csls: bool) -> object:
    """
    Create one object with the data to create the pandas dataframe.
    Parameters
    ----------
    method BootEA or RDGCN
    dataset the name of the dataset
    measure hits1, hits5, mrr and so on
    results list of results obtained in the k-folds
    csls True or false

    Returns
    -------
    An object with the correct parameters in order to populate the dataframe which will be output.
    """
    return {
        "method": method,
        "dataset": dataset,
        "measure": measure,
        "mean": np.array(results).mean(),
        "std": np.array(results).std(),
        "csls": csls
    }


if __name__=='__main__':
    hits1 = []
    hits5 = []
    hits10 = []
    hits50 = []
    hits1_csls = []
    hits5_csls = []
    hits10_csls = []
    hits50_csls = []
    mr = []
    mrr = []
    mr_csls = []
    mrr_csls = []

    string_search = "accurate results: hits@[1, 5, 10, 50] = "
    string_search_csls = "accurate results with csls: csls=\d+, hits@\[1, 5, 10, 50\] = "

    # Parse the files and load the results.
    for f_name in args.files:
        with open(f_name, 'r') as f:
            log = f.read()
            # Read the results for hits1, hits5, hits10, hits50.
            hits = [float(num) for num in log.split(string_search)[-1].split(
                   '[')[1].split(']')[0].split()]
            mr.append(float(log.split(string_search)[-1].split("mr = ")[1].split(',')[0]))
            mrr.append(float(log.split(string_search)[-1].split("mrr = ")[1].split(',')[0]))

            hits1.append(hits[0])
            hits5.append(hits[1])
            hits10.append(hits[2])
            hits50.append(hits[3])

            # Only RDGCA has csls results.
            hits_cls = [float(num) for num in re.split(
                       string_search_csls, log)[-1].split(
                       '[')[1].split(']')[0].split()]

            mr_csls.append(float(re.split(
                string_search_csls,
                log)[-1].split("mr = ")[1].split(',')[0]))
            mrr_csls.append(float(re.split(
                string_search_csls,
                log)[-1].split("mrr = ")[1].split(',')[0]))

            hits1_csls.append(hits_cls[0])
            hits5_csls.append(hits_cls[1])
            hits10_csls.append(hits_cls[2])
            hits50_csls.append(hits_cls[3])

    # This list will keep the rows of the dataframe.
    df_data = []

    # Log the result
    print("Method: {}".format(args.method))
    print("Dataset: {}".format(args.dataset))
    print("Input files:\n{}".format("\n".join(args.files)))

    df_data.append(create_df_data_entry(args.method, args.dataset, "hits1", hits1, False))
    df_data.append(create_df_data_entry(args.method, args.dataset, "hits5", hits5, False))
    df_data.append(create_df_data_entry(args.method, args.dataset, "hits10", hits10, False))
    df_data.append(create_df_data_entry(args.method, args.dataset, "hits50", hits50, False))
    df_data.append(create_df_data_entry(args.method, args.dataset, "mr", mr, False))
    df_data.append(create_df_data_entry(args.method, args.dataset, "mrr", mrr, False))

    df_data.append(create_df_data_entry(args.method, args.dataset, "hits1", hits1_csls, True))
    df_data.append(create_df_data_entry(args.method, args.dataset, "hits5", hits5_csls, True))
    df_data.append(create_df_data_entry(args.method, args.dataset, "hits10", hits10_csls, True))
    df_data.append(create_df_data_entry(args.method, args.dataset, "hits50", hits50_csls, True))
    df_data.append(create_df_data_entry(args.method, args.dataset, "mr", mr_csls, True))
    df_data.append(create_df_data_entry(args.method, args.dataset, "mrr", mrr_csls, True))

    df = pd.DataFrame(df_data)

    # Print to excel or to std output
    if args.output_format == 'xls':
        df.to_excel(args.output_file)
    else:
        df.to_csv(args.output_file)

