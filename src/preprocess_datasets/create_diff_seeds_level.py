import argparse
import os
import random
import numpy as np


def create_folds_inc_seed(dataset, k, division, percentage_step, do_one_percent, do_zero_percent):
    with open(dataset + "/ent_links") as f:
        links = set(f.readlines())
    test, train, valid = division.split("/")
    test = int(test) / 10
    train = int(train) / 10
    valid = int(valid) / 10
    print("Chosen test: {}, Chosen train: {}, Chosen validation: {}".format(test, train, valid))
    split_folder = division.replace("/", "") + "_" + str(k) + "folds"
    if not os.path.isdir(dataset + "/" + split_folder):
        os.mkdir(dataset + "/" + split_folder)
    k_dict = {}
    k_dict_increasing_train = {}
    for k_split in range(1, k + 1):
        test_links = set(random.sample(links, int(test * len(links))))
        valid_train_links = links - test_links
        train_links = set(random.sample(valid_train_links, int(train * len(links))))
        valid_links = valid_train_links - train_links
        k_dict[k_split] = (test_links, train_links, valid_links)
        k_dict_increasing_train[k_split] = set()
    seed_list = list(np.arange(percentage_step, train + percentage_step, percentage_step))
    if do_zero_percent:
        print("Working on seed 0")
        seed_folder = dataset + "/" + split_folder + "/0_seed"
        if not os.path.isdir(seed_folder):
            os.mkdir(seed_folder)
        for k_split in range(1, k + 1):
            k_folder = seed_folder + "/" + str(k_split)
            if not os.path.isdir(k_folder):
                os.mkdir(k_folder)
            test_links = k_dict[k_split][0] | k_dict[k_split][1]
            train_links = []
            with open(k_folder + "/test_links", "w") as f:
                f.writelines(test_links)
            with open(k_folder + "/train_links", "w") as f:
                f.writelines(train_links)
            with open(k_folder + "/valid_links", "w") as f:
                f.writelines(k_dict[k_split][2])
            print("Written fold {}!".format(k_split))
    if do_one_percent:
        seed_list.insert(0, 0.01)
    for seed in seed_list:
        print("Working on seed {}".format(seed))
        seed_folder = dataset + "/" + split_folder + "/" + str(int(seed * 100)) + "_seed"
        if not os.path.isdir(seed_folder):
            os.mkdir(seed_folder)
        for k_split in range(1, k + 1):
            k_folder = seed_folder + "/" + str(k_split)
            if not os.path.isdir(k_folder):
                os.mkdir(k_folder)
            new_train = int(seed * len(links)) - len(k_dict_increasing_train[k_split])
            poss_train = k_dict[k_split][1] - k_dict_increasing_train[k_split]
            train_links = k_dict_increasing_train[k_split] | set(random.sample(poss_train, new_train))
            test_links = k_dict[k_split][0] | (k_dict[k_split][1] - train_links)
            #print("Train size: {}, Test size: {}".format(len(train_links), len(test_links)))
            k_dict_increasing_train[k_split] = train_links
            with open(k_folder + "/test_links", "w") as f:
                f.writelines(test_links)
            with open(k_folder + "/train_links", "w") as f:
                f.writelines(train_links)
            with open(k_folder + "/valid_links", "w") as f:
                f.writelines(k_dict[k_split][2])
            print("Written fold {}!".format(k_split))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create k random folds from the given datasets '
                                                 'with the selected division')
    parser.add_argument("--dataset", type=str, help="Path to dataset folder")
    parser.add_argument("--k", type=int, help="Number of folds")
    parser.add_argument("--division", type=str, help="Division, with format test/train/validation")
    parser.add_argument("--percentage_step", type=float, help="How to increase percentage")
    parser.add_argument("--do_one_percent", action="store_true", default=False)
    parser.add_argument("--do_zero_percent", action="store_true", default=False)
    args = parser.parse_args()
    create_folds_inc_seed(args.dataset, args.k, args.division, args.percentage_step, args.do_one_percent, args.do_zero_percent)
