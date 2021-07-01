import argparse
import os
import json


def tuning(parameter, lists_values, original_args, method, dataset, dataset_division, gpu):
    with open(original_args) as f:
        old_args = json.load(f)
    hyper_dict = {}
    for hyper in lists_values:
        old_args[parameter] = int(float(hyper)) if int(float(hyper)) == float(hyper) else float(hyper)
        new_args_path = "args_hyper/args_{}_{}_{}_{}.json".format(method, dataset, parameter, hyper.replace(".", "_"))
        with open(new_args_path, 'w') as json_file:
            json.dump(old_args, json_file)
        log_file = "args_hyper/args_{}_{}_{}_{}.log".format(method, dataset, parameter, hyper.replace(".", "_"))
        command = "python3 -u main_from_args.py {} {} {} {} 2>&1 | tee {}" \
            .format(new_args_path, dataset, dataset_division, gpu, log_file)
        os.system(command)
        with open(log_file) as log:
            log_str = log.read()
            f1 = float(log_str.split("F1: ")[-2].split("\n")[0])
            precision = float(log_str.split("Precision: ")[-2].split("\n")[0])
            recall = float(log_str.split("Recall: ")[-2].split("\n")[0])
            hyper_dict[hyper] = (precision, recall, f1)
            print("Hyper: {}\n\tPrecision: {}\n\tRecall: {}\n\tF1: {}".format(hyper, precision, recall, f1))
    max_f1 = 0
    best_hyper = None
    for (hyper, stats) in hyper_dict.items():
        if stats[2] > max_f1:
            max_f1 = stats[2]
            best_hyper = hyper
    print("Best Hyper: {}\n\tPrecision: {}\n\tRecall: {}\n\tF1: {}"
          .format(best_hyper, hyper_dict[best_hyper][0], hyper_dict[best_hyper][1], max_f1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hyper-Parameters tuning')
    parser.add_argument('--parameter', type=str, help="Parameter to test")
    parser.add_argument(
        "--list_values",
        nargs="+", type=str,
        help='List of values to test.'
    )
    parser.add_argument("--original_args", type=str, help="Path to original file from where to load params")
    parser.add_argument("--method", type=str, help="Method to tune")
    parser.add_argument("--dataset", type=str, help="Dataset to use")
    parser.add_argument("--dataset_division", type=str, help="Dataset fold division")
    parser.add_argument("--gpu", type=str, help="GPU to use")
    args = parser.parse_args()
    tuning(args.parameter, args.list_values, args.original_args, args.method, args.dataset, args.dataset_division,
           args.gpu)
