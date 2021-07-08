import argparse
import os
from tqdm import tqdm

parser = argparse.ArgumentParser(
        description="Checks that anonymous dataset is coherent with its non anonymous counterpart.")
parser.add_argument(
    "--dataset_non_anon_folder",
    type=str,
    required=True,
    help='root folder of the non anonymous dataset'
)

parser.add_argument(
    "--dataset_anon_folder",
    type=str,
    help="root folder of the anonymous dataset",
    required=True
)
args = parser.parse_args()


def RepresentsInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def check_fold(fold_name: str, map_anon_to_no, map_no_to_anon, non_anon_dataset_folder, anon_dataset_folder):
    # TODO: Still check valid train
    for fold_dir in os.listdir("{}/{}".format(anon_dataset_folder, fold_name)):
        if not RepresentsInt(fold_dir):
            print(fold_dir)
            fold_dir = "/"
        for file in ['test_links', 'train_links', 'valid_links']:
            with open("{}/{}/{}/{}".format(non_anon_dataset_folder, fold_name, fold_dir, file)) as f:
                with open("{}/{}/{}/{}".format(anon_dataset_folder, fold_name, fold_dir, file)) as f2:
                    for l1, l2 in tqdm(zip(f, f2)):
                        e_non_anon_1, e_non_anon_2 = l1.rstrip("\n").split('\t')
                        e_anon_1, e_anon_2 = l2.rstrip("\n").split('\t')
                        if e_anon_1 not in map_anon_to_no:
                            print(e_anon_1, "does not appear in map_anon_to_no")
                            return False
                        if map_anon_to_no[e_anon_1] != e_non_anon_1:
                            print("In {} the following pair is non-coherent".format(
                                "{}/{}/{}/{}".format(non_anon_dataset_folder, fold_name, fold_dir, file)
                            ))
                            print(e_anon_1, e_non_anon_1)
                            return False
                        if e_non_anon_1 not in map_no_to_anon:
                            print(e_non_anon_1, "does not appear in map_no_to_anon")
                            return False
                        if map_no_to_anon[e_non_anon_1] != e_anon_1:
                            print("In {} the following pair is non-coherent".format(
                                "{}/{}/{}/{}".format(non_anon_dataset_folder, fold_name, fold_dir, file)
                            ))
                            print(e_anon_1, e_non_anon_1)
                            return False
                        if e_anon_2 not in map_anon_to_no:
                            print(e_anon_2, "does not appear in map_anon_to_no")
                            return False
                        if map_anon_to_no[e_anon_2] != e_non_anon_2:
                            print("In {} the following pair is non-coherent".format(
                                "{}/{}/{}/{}".format(non_anon_dataset_folder, fold_name, fold_dir, file)
                            ))
                            print(e_anon_2, e_non_anon_2)
                            return False
                        if e_non_anon_2 not in map_no_to_anon:
                            print(e_non_anon_2, "does not appear in map_no_to_anon")
                            return False
                        if map_no_to_anon[e_non_anon_2] != e_anon_2:
                            print("In {} the following pair is non-coherent".format(
                                "{}/{}/{}/".format(non_anon_dataset_folder, fold_name, fold_dir, file)
                            ))
                            print(e_anon_2, e_non_anon_2)
                            return False
        print("{}/{}/{} are OK!".format(fold_name, fold_dir, file))
    return True


def check_attr(non_anon_dataset_folder, anon_dataset_folder, map_no_to_anon, map_anon_to_no, triple_file_num: str):
    with open("{}/attr_triples_{}".format(non_anon_dataset_folder, triple_file_num)) as f:
        with open("{}/attr_triples_{}".format(anon_dataset_folder, triple_file_num)) as f2:
            for l1, l2 in tqdm(zip(f, f2)):
                e_non_anon, a, v = l1.rstrip("\n").split('\t')
                e_anon, a2, v2 = l2.rstrip("\n").split('\t')
                if e_non_anon not in map_no_to_anon:
                    map_no_to_anon[e_non_anon] = e_anon
                if e_anon not in map_anon_to_no:
                    map_anon_to_no[e_anon] = e_non_anon
                if map_anon_to_no[e_anon] != e_non_anon:
                    print("In attr_triples, this pair is non-coherent")
                    print(e_non_anon, e_anon)
                    return False
                if map_no_to_anon[e_non_anon] != e_anon:
                    print("In attr_triples, this pair is non-coherent")
                    print(e_non_anon, e_anon)
                    return False
    return True


def check_rel(non_anon_dataset_folder, anon_dataset_folder, map_no_to_anon, map_anon_to_no, rel_file_num):
    # Check rel triples
    with open("{}/rel_triples_{}".format(non_anon_dataset_folder, rel_file_num)) as f:
        with open("{}/rel_triples_{}".format(anon_dataset_folder, rel_file_num)) as f2:
            for l1, l2 in tqdm(zip(f, f2)):
                e_non_anon_1, r, e_non_anon_2 = l1.rstrip("\n").split('\t')
                e_anon_1, r2, e_anon_2 = l2.rstrip("\n").split('\t')
                if e_non_anon_1 not in map_no_to_anon:
                    map_no_to_anon[e_non_anon_1] = e_anon_1
                if e_anon_1 not in map_anon_to_no:
                    map_anon_to_no[e_anon_1] = e_non_anon_1
                if e_non_anon_2 not in map_no_to_anon:
                    map_no_to_anon[e_non_anon_2] = e_anon_2
                if e_anon_2 not in map_anon_to_no:
                    map_anon_to_no[e_anon_2] = e_non_anon_2
                if map_anon_to_no[e_anon_1] != e_non_anon_1:
                    print("In rel_triples, this pair is non-coherent")
                    print(e_non_anon_1, e_anon_1)
                    return False
                if map_no_to_anon[e_non_anon_1] != e_anon_1:
                    print("In rel_triples, this pair is non-coherent")
                    print(e_non_anon_1, e_anon_1)
                    return False
                if map_no_to_anon[e_non_anon_2] != e_anon_2:
                    print("In rel_triples, this pair is non-coherent")
                    print(e_non_anon_2, e_anon_2)
                    return False
                if map_anon_to_no[e_anon_2] != e_non_anon_2:
                    print("In rel_triples, this pair is non-coherent")
                    print(e_non_anon_2, e_anon_2)
                    return False
    return True


def check_dataset(non_anon_dataset_folder: str, anon_dataset_folder: str) -> bool:
    """
    Given two datasets, one anonymous and the non-anonymous counterpart,
    checks whether the two datasets are coherent.
    In particular, every entity should be mapped to one anonymous id.
    Parameters
    ----------
    non_anon_dataset_folder
    anon_dataset_folder

    Returns
    -------

    """
    map_no_to_anon = {}
    map_anon_to_no = {}

    if not check_attr(non_anon_dataset_folder, anon_dataset_folder, map_no_to_anon, map_anon_to_no, "1"):
        return False
    print("attr triples 1 OK!")
    if not check_attr(non_anon_dataset_folder, anon_dataset_folder, map_no_to_anon, map_anon_to_no, "2"):
        return False
    print("attr triples 2 OK!")

    if not check_rel(non_anon_dataset_folder, anon_dataset_folder, map_no_to_anon, map_anon_to_no, "1"):
        return False
    print("rel triples 1 OK!")
    if not check_rel(non_anon_dataset_folder, anon_dataset_folder, map_no_to_anon, map_anon_to_no, "2"):
        return False
    print("rel triples 2 OK!")
    print("The size of map_no_to_anon is", len(map_no_to_anon))

    print("The size of map_anon_to_no is", len(map_anon_to_no))
    if "721_5folds" in os.listdir(anon_dataset_folder) and "721_5folds" in os.listdir(non_anon_dataset_folder):
        if not check_fold("721_5folds", map_anon_to_no, map_no_to_anon, non_anon_dataset_folder, anon_dataset_folder):
            return False
    print("Fold 721_5folds OK!")
    if "631_5folds" in os.listdir(anon_dataset_folder) and "631_5folds" in os.listdir(non_anon_dataset_folder):
        # Increasing seed experiments
        for seed in os.listdir(anon_dataset_folder + "/631_5folds"):
            if not check_fold("631_5folds/{}".format(seed), map_anon_to_no, map_no_to_anon, non_anon_dataset_folder, anon_dataset_folder):
                return False
    print("Fold 721_5folds OK!")
    # if "631" in os.listdir(anon_dataset_folder) and "631" in os.listdir(non_anon_dataset_folder):
    #     if not check_fold("631", map_anon_to_no, map_no_to_anon, non_anon_dataset_folder, anon_dataset_folder):
    #         return False
    # print("Fold 631 OK!")
    # if "721_1folds" in os.listdir(anon_dataset_folder) and "721_1folds" in os.listdir(non_anon_dataset_folder):
    #     if not check_fold("721_1folds", map_anon_to_no, map_no_to_anon, non_anon_dataset_folder, anon_dataset_folder):
    #         return False
    # print("Fold 721_1folds OK!")
    return True


if __name__ == "__main__":
    correct = check_dataset(args.dataset_non_anon_folder, args.dataset_anon_folder)
    if correct:
        print("Datasets {} is truly anonymous for {}".format(args.dataset_anon_folder, args.dataset_non_anon_folder))
        print("SUCCESS!")
    else:
        print("THE DATASETS ARE WRONG!!")
