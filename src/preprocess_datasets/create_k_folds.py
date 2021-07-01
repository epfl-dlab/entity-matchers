import argparse
import os
import random


def create_folds(dataset, k, division):
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
    for k_split in range(1, k + 1):
        k_folder = dataset + "/" + split_folder + "/" + str(k_split)
        if not os.path.isdir(k_folder):
            os.mkdir(k_folder)
        test_links = set(random.sample(links, int(test * len(links))))
        valid_train_links = links - test_links
        train_links = set(random.sample(valid_train_links, int(train * len(links))))
        valid_links = valid_train_links - train_links
        with open(k_folder + "/test_links", "w") as f:
            f.writelines(test_links)
        with open(k_folder + "/train_links", "w") as f:
            f.writelines(train_links)
        with open(k_folder + "/valid_links", "w") as f:
            f.writelines(valid_links)
        print("Written fold {}!".format(k_split))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create k random folds from the given datasets '
                                                 'with the selected division')
    parser.add_argument("--dataset", type=str, help="Path to dataset folder")
    parser.add_argument("--k", type=int, help="Number of folds")
    parser.add_argument("--division", type=str, help="Division, with format test/train/validation")
    args = parser.parse_args()
    create_folds(args.dataset, args.k, args.division)
