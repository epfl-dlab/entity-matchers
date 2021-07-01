import argparse
import os
import json
import pickle

def tuning(parameters, dataset, fold_folder, gpu):
    with open(parameters) as f:
        parameters_dict = json.load(f)
    print("Params to test:", parameters_dict)
    best_params = {}
    command_base = "python3 -u transedge_ea.py --data_dir {} --fold_folder {} --gpu {}"\
        .format(dataset, fold_folder, gpu)
    for (param, list_values) in parameters_dict.items():
        print("\nStarting the test for:", param)
        hyper_dict = {}
        command_so_far = command_base
        for (best_param, best_val) in best_params.items():
            command_so_far = command_so_far + " --{} {}".format(best_param, best_val)
        for hyper in list_values:
            log_file = "args_hyper/args_TransEdge_{}_{}_{}.log"\
                .format(dataset.split("/")[-1], param, str(hyper).replace(".", "_"))
            print("\tTesting", hyper)
            print("\tLog File:", log_file)
            command = command_so_far + " --{} {} > {} 2>&1".format(param, hyper, log_file)
            os.system(command)
            with open(log_file) as log:
                log_str = log.read()
                try:
                    f1 = float(log_str.split("F1: ")[-1].split("\n")[0])
                except ValueError:
                    print("Something bad happened during execution! Skipping the parameter...")
                    continue
                precision = float(log_str.split("Precision: ")[-1].split("\n")[0])
                recall = float(log_str.split("Recall: ")[-1].split("\n")[0])
                hyper_dict[hyper] = (precision, recall, f1)
                print("Param: {}\nValue:{}\n\tPrecision: {}\n\tRecall: {}\n\tF1: {}"
                      .format(param, hyper, precision, recall, f1))
        max_f1 = 0
        best_hyper = None
        for (hyper, stats) in hyper_dict.items():
            if stats[2] > max_f1:
                max_f1 = stats[2]
                best_hyper = hyper
        best_params[param] = best_hyper
        print("Param: {}\nBest Hyper: {}\n\tPrecision: {}\n\tRecall: {}\n\tF1: {}"
              .format(param, best_hyper, hyper_dict[best_hyper][0], hyper_dict[best_hyper][1], max_f1))
    pickle_name = "args_best/best_TransEdge_{}.pkl".format(dataset.split("/")[-1])
    with open(pickle_name, "wb") as f:
        pickle.dump(best_params, f)
    print("Saved best args in", pickle_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hyper-Parameters tuning')
    parser.add_argument('--json_params', type=str, help="Path to json of dict of params to test")
    parser.add_argument("--dataset", type=str, help="Dataset to use")
    parser.add_argument("--fold_folder", type=str, help="Folder to use")
    parser.add_argument("--gpu", type=str, help="GPU to use", default="0")
    args = parser.parse_args()
    if "args_hyper" not in os.listdir("."):
        os.mkdir("args_hyper")
    if "args_best" not in os.listdir("."):
        os.mkdir("args_best")
    tuning(args.json_params, args.dataset, args.fold_folder, args.gpu)
