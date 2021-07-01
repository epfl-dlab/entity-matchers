import argparse
import os
from utils import *
import numpy as np
import json
import copy


def run_paris_experiment(root_dataset, dataset, dataset_division, out_folder):
    precisions = []
    recalls = []
    f1s = []
    train_times = []
    test_times = []

    dataset_in = root_dataset + "/" + dataset + "/"
    kg1_file = "kg1.nt"
    kg2_file = "kg2.nt"
    for fold in os.listdir(dataset_in + dataset_division):
        start_time = time.time()
        create_nt(dataset_in, dataset_division, fold, kg1_file, kg2_file)
        paris_out_folder = run_paris(out_folder, dataset, kg1_file, kg2_file)
        train_time = time.time() - start_time
        start_time = time.time()
        precision, recall, f1 = evaluate_paris(paris_out_folder, dataset_in, dataset_division, fold)
        test_time = time.time() - start_time

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        train_times.append(train_time)
        test_times.append(test_time)
    os.system("rm kg1.nt")
    os.system("rm kg2.nt")

    precisions = np.array(precisions)
    recalls = np.array(recalls)
    f1s = np.array(f1s)
    train_times = np.array(train_times)
    test_times = np.array(test_times)

    print("precisions: ", precisions)
    print("recalls: ", recalls)
    print("f1s: ", f1s)
    print("Train times: ", train_times)
    print("Test times: ", test_times)

    print("precisions:\n\tavg: {}\n\tstd: {}".format(precisions.mean(), precisions.std()))
    print("recalls:\n\tavg: {}\n\tstd: {}".format(recalls.mean(), recalls.std()))
    print("f1s:\n\tavg: {}\n\tstd: {}".format(f1s.mean(), f1s.std()))
    print("Train times:\n\tavg: {}\n\tstd: {}".format(train_times.mean(), train_times.std()))
    print("Test times:\n\tavg: {}\n\tstd: {}".format(test_times.mean(), test_times.std()))


def run_embedding_experiment(root_dataset, args, out_folder, dataset, dataset_division, gpu, method, main_embeds):
    precisions_no_csls = []
    precisions_csls = []
    recalls_no_csls = []
    recalls_csls = []
    f1s_no_csls = []
    f1s_csls = []
    list_train_times = []  # Time specified in seconds
    list_test_times = []  # Time specified in seconds

    dataset_in = root_dataset + "/" + dataset + "/"
    for fold in os.listdir(dataset_in + dataset_division):
        # Read the json file with the configuration, and substitute temporarily the dataset location
        with open(args) as json_file:
            json_data = json.load(json_file)
            old_json_data = copy.deepcopy(json_data)

        json_data['training_data'] = root_dataset + '/'
        json_data['output'] = out_folder + '/'

        # Re-write the json with the correct information
        with open(args, 'w') as outfile:
            json.dump(json_data, outfile)
        current_time = time.localtime()
        log_file = out_folder + '/' + "args_{}_{}_{}_{}_{}_{}_{}.log".format(method,
                                                                             dataset,
                                                                             current_time.tm_mon,
                                                                             current_time.tm_mday,
                                                                             current_time.tm_hour,
                                                                             current_time.tm_min,
                                                                             current_time.tm_sec)
        command = "python3 -u {} {} {} {} {} > {}" \
            .format(main_embeds, args, dataset, dataset_division + '/' + fold + '/', gpu, log_file)
        os.system(command)

        # Read the statistics: both with csls and without csls

        precision_no_csls, precision_csls, recall_no_csls, \
        recall_csls, f1_no_csls, f1_csls, train_time_seconds, test_time_seconds = parse_stats_from_log(log_file)

        precisions_no_csls.append(precision_no_csls)
        recalls_no_csls.append(recall_no_csls)
        f1s_no_csls.append(f1_no_csls)
        precisions_csls.append(precision_csls)
        recalls_csls.append(recall_csls)
        f1s_csls.append(f1_csls)
        list_train_times.append(train_time_seconds)
        list_test_times.append(test_time_seconds)

        # Restore the old json file.
        with open(args, 'w') as outfile:
            json.dump(old_json_data, outfile)
    precisions_no_csls = np.array(precisions_no_csls)
    precisions_csls = np.array(precisions_csls)
    recalls_no_csls = np.array(recalls_no_csls)
    recalls_csls = np.array(recalls_csls)
    f1s_no_csls = np.array(f1s_no_csls)
    f1s_csls = np.array(f1s_csls)
    list_train_times = np.array(list_train_times)
    list_test_times = np.array(list_test_times)

    print("precisions no csls: ", precisions_no_csls)
    print("recalls no csls: ", recalls_no_csls)
    print("f1s no csls: ", f1s_no_csls)
    print("precisions csls: ", precisions_csls)
    print("recalls csls: ", recalls_csls)
    print("f1s csls: ", f1s_csls)
    print('Train times (seconds): ', list_train_times)
    print('Test times (seconds): ', list_test_times)

    print("precisions no csls:\n\tavg: {}\n\tstd: {}".format(precisions_no_csls.mean(), precisions_no_csls.std()))
    print("recalls no csls:\n\tavg: {}\n\tstd: {}".format(recalls_no_csls.mean(), recalls_no_csls.std()))
    print("f1s no csls:\n\tavg: {}\n\tstd: {}".format(f1s_no_csls.mean(), f1s_no_csls.std()))
    print("precisions csls:\n\tavg: {}\n\tstd: {}".format(precisions_csls.mean(), precisions_csls.std()))
    print("recalls csls:\n\tavg: {}\n\tstd: {}".format(recalls_csls.mean(), recalls_csls.std()))
    print("f1s csls:\n\tavg: {}\n\tstd: {}".format(f1s_csls.mean(), f1s_csls.std()))
    print("Train times:\n\tavg: {}\n\tstd: {}".format(list_train_times.mean(), list_train_times.std()))
    print("Test times:\n\tavg: {}\n\tstd: {}".format(list_test_times.mean(), list_test_times.std()))


def run_exps(method, args, root_dataset, dataset, dataset_division, gpu, out_folder, main_embeds):
    if method == "PARIS":
        run_paris_experiment(root_dataset, dataset, dataset_division, out_folder)

    elif method == 'RDGCN' or method == 'BOOTEA':
        run_embedding_experiment(root_dataset, args, out_folder, dataset, dataset_division, gpu, method, main_embeds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run one experiment with a given dataset")
    parser.add_argument("--method", help="Method to use", choices=["PARIS", "RDGCN", "BOOTEA"], type=str)
    parser.add_argument("--args", type=str, help="Path to file from where to load params (only for embeddings methods)")
    parser.add_argument("--root_dataset", type=str, help="Path to dataset root folder (no slash in the end)")
    parser.add_argument("--dataset", type=str, help="Dataset to use (no slash in the end)")
    parser.add_argument("--dataset_division", type=str, help="Dataset fold division (no slash in the end)")
    parser.add_argument("--gpu", type=str, help="GPU to use (only for embeddings methods)")
    parser.add_argument("--out_folder", type=str, help="Root folder for output (no slash in the end)")
    parser.add_argument("--main_embeds", type=str, help="Path to main script for embeddings method")
    args = parser.parse_args()
    run_exps(args.method, args.args, args.root_dataset, args.dataset,
             args.dataset_division, args.gpu, args.out_folder, args.main_embeds)
