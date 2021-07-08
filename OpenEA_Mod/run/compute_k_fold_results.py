import sys
import numpy as np
import argparse
import re

# Parse command line arguments (the names of the files to parse).
parser = argparse.ArgumentParser(
    description='Parse the log files provided as input and compute average and'
                ' standard deviation for the metrics hit1, hit5, '
                'hit10, hit50, mr, mrr among the files provided as input.'
                'If you want to redirect the printed results to a file, you can use the > operator')

parser.add_argument('-m', '--method',
                    help='name of the method either RDGCN or BootEA (careful with the spelling)',
                    required=True,
                    choices=['BootEA', 'RDGCN'])

parser.add_argument('-d', '--dataset',
                    help='name of the dataset, for example D_W_15K_V1',
                    required=True)

parser.add_argument('-f', '--files',
                    nargs='+',
                    help='list of log files names',
                    required=True)


args = parser.parse_args()


def list_and_stats(name_of_list: str, my_list: list) -> str:
    """
    Return a list with the list and its stats
    Parameters
    ----------
    name_of_list: name to be displayed for the list
    my_list: the list itself

    Returns
    -------
    a string with the list, the average and the standard deviation
    """
    return "{name_of_list}:\n\t{list}\n\tavg: {avg}\n\tstd: {std}".format(
        name_of_list=name_of_list,
        list=my_list,
        avg=np.array(my_list).mean(),
        std=np.array(my_list).std()
    )


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

    # Log the result
    print("Method: {}".format(args.method))
    print("Dataset: {}".format(args.dataset))
    print("Input files:\n{}".format("\n".join(args.files)))

    print("\n#############################################\n"
          "RESULTS:\n"
          "#############################################\n")
    print(list_and_stats("hits1", hits1))
    print(list_and_stats("hits5", hits5))
    print(list_and_stats("hits10", hits10))
    print(list_and_stats("hits50", hits50))
    print(list_and_stats("mr", mr))
    print(list_and_stats("mrr", mrr))

    print("\n#############################################\n"
          "RESULTS CSLS:\n"
          "#############################################\n")
    print(list_and_stats("hits1_csls", hits1_csls))
    print(list_and_stats("hits5_csls", hits5_csls))
    print(list_and_stats("hits10_csls", hits10_csls))
    print(list_and_stats("hits50_csls", hits50_csls))
    print(list_and_stats("mr_csls", mr_csls))
    print(list_and_stats("mrr_csls", mrr_csls))



