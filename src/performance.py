import pandas as pd
import numpy as np
import os

def precision(same_list: list, res_list: list):
    """
    Compute the precision, given the two result.

    Args:
        same_list (list): Lines from the ground truth, elaborated to be compared with the PARIS result
        res_list (list): PARIS result, elaborated to be compared with the ground truth

    Returns:
        precision(float): Precision
    """
    if len(res_list) == 0:
        return 0
    # Use precision definition from Information Retrieval
    same_set = set(same_list)
    res_set = set(res_list)
    precision = len(same_set.intersection(res_set)) / len(
        res_set
    )  # Divide by the size of the found result
    return precision


def recall(same_list, res_list):
    """
    Compute the recall, given the two result

    Args:
        same_list (list): Lines from the ground truth, elaborated to be compared with the PARIS result
        res_list (list): PARIS result, elaborated to be compared with the ground truth

    Returns:
        recall(float): Recall
    """
    if len(res_list) == 0:
        return 0
    # Use recall definition from Information Retrieval
    same_set = set(same_list)
    res_set = set(res_list)
    recall = len(same_set.intersection(res_set)) / len(
        same_set
    )  # Divide by the size of the truth
    return recall


def f1_score(precision: float, recall: float):
    """
    Compute the F1 score, given Precision and recall

    Args:
        precision (float): Precision at the given iteration
        recall (float): Recall at the given iteration

    Returns:
        f1(float): F1 score
    """
    if precision == 0 or recall == 0:
        return 0
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def check_result(folder:str, dataset_type: str, out_path: str, dataset_division: str = "721_5fold", fold_num: str = None):
  
    run = 0
    res_list = []
    while True:
        full_path = out_path + "/{run}_eqv.tsv".format(run=run)
        if os.path.exists(full_path):
            # PARIS create an empty file at the last_iter+1. If we encountered it, we can break
            if os.stat(full_path).st_size == 0:
                break

            # Get PARIS result from the .tsv and elaborate it a bit to be compared with the same_list
            res_list = pd.read_csv(
                full_path, delimiter="\t", header=None, usecols=[0, 1]
            ).values.tolist()
            res_list = [" ".join(x) for x in res_list]
            dbpedia = False
            for res in res_list:
                if "dbp:" in res:
                    dbpedia = True
            if dbpedia:
                res_list = [res for res in res_list if "dbp:" in res]
        run+=1
    
    if dataset_type=="OpenEA_dataset":
        same_full = open(folder + "ent_links", "r")
        same_list = same_full.readlines()
        same_list = [
            same.replace("http://dbpedia.org/", "dbp:").replace("\t", " ").rstrip()
            for same in same_list
        ]
        same_full.close()

        prec = precision(same_list, res_list)
        rec = recall(same_list, res_list)
        f1 = f1_score(prec, rec)
        print("Ground truth precision:", prec)
        print("Ground truth recall:", rec)
        print("Ground truth F1 score:", f1)

        if fold_num!=None:
            print("Fold", fold_num)
            train_fold = open(folder + dataset_division + "/" + fold_num + "/" + "train_links")
            test_fold = open(folder + dataset_division + "/" + fold_num + "/" + "test_links")
            valid_fold = open(folder + dataset_division + "/" + fold_num + "/" + "valid_links")
            train_list = train_fold.readlines()
            train_list = [
                train.replace("http://dbpedia.org/", "dbp:").replace("\t", " ").rstrip()
                for train in train_list
            ]
            train_fold.close()
            test_list = test_fold.readlines()
            test_list = [
                test.replace("http://dbpedia.org/", "dbp:").replace("\t", " ").rstrip()
                for test in test_list
            ]
            test_fold.close()
            valid_list = valid_fold.readlines()
            valid_list = [
                valid.replace("http://dbpedia.org/", "dbp:").replace("\t", " ").rstrip()
                for valid in valid_list
            ]
            valid_fold.close()
            res_list_test = list(set(res_list).difference(set(train_list)).difference(set(valid_list)))
            prec_test = precision(test_list, res_list_test)
            rec_test = recall(test_list, res_list_test)
            f1_test = f1_score(prec_test, rec_test)
            print("Test links precision:", prec_test)
            print("Test links recall:", rec_test)
            print("Test links F1 score:", f1_test)
            res_list_valid = list(set(res_list).difference(set(train_list)).difference(set(test_list)))
            prec_valid = precision(valid_list, res_list_valid)
            rec_valid = recall(valid_list, res_list_valid)
            f1_valid = f1_score(prec_valid, rec_valid)
            print("Valid links precision:", prec_valid)
            print("Valid links recall:", rec_valid)
            print("Valid links F1 score:", f1_valid)
            return prec, rec, f1, prec_test, rec_test, f1_test, prec_valid, rec_valid, f1_valid
    else:
        # TODO: do something for not OpenEA_datasets
        return



