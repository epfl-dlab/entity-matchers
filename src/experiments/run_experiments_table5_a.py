"""
This script will reproduce the results in table 5-a of the paper. Note that each dataset sub-table will be stored into
a dataset-specific .csv file into the directory ./output/output_table5_a. You can also find the detailed logs for each
run and experiment into ./output/output_table5_a/log_folds and ./output/output_table5_a/log_exp.

This script assume that you created the environments to run all the experiments as specified in our README.md.
You need to have one environment entity_match for all the methods but BERT-INT and one called bert-int for BERT-INT.

Note that for RDGCN experiments you need to modify the args under args_best/rdgcn_args_{dataset} and modify the
"word_embed" key to point to the path where you downloaded the embeddings
(as specified into https://github.com/epfl-dlab/entity-matchers#reproduction-of-results)

You need to provide the path to the folder containing all the datasets used in the paper, downloaded as specified into
our GitHub README.md. Please do not add any slash at the end of the path.
You also need to specify a GPU id (0 if you only have one GPU) to use. If you don't specify any id, the experiments will
be run on CPU (requiring more time). We highly recommend the use of a GPU
"""

import argparse
import os
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproduce results in Table 5-a")
    parser.add_argument("--root_dataset", type=str,
                        help="Path to the root folder containing the downloaded datasets (no slash)")
    parser.add_argument("--gpu", type=str, help="GPU id to use (if present)")
    args = parser.parse_args()

    methods = ["RDGCN", "BOOTEA", "TRANSEDGE", "BERT-INT", "PARIS"]
    datasets = ["DB-YG-15K", "DB-WD-15K", "DB-YG-100K", "DB-WD-100K"]
    output_base = "./output"
    output_dir = f"{output_base}/output_table5_a"
    output_dir_log_folds = f"{output_dir}/log_folds"
    output_dir_log_exp = f"{output_dir}/log_exp"
    for directory in [output_base, output_dir, output_dir_log_exp, output_dir_log_folds]:
        if not os.path.exists(directory):
            os.mkdir(directory)
    for method in methods:
        output_dir_log_fold_method = f"{output_dir_log_folds}/{method}"
        if not os.path.exists(output_dir_log_fold_method):
            os.mkdir(output_dir_log_fold_method)
    gpu = args.gpu if args.gpu else "CPU"
    arg_map = {}
    for method in methods:
        arg_map[method] = {}
        for dataset in datasets:
            if "DB" in dataset:
                dataset_arg = dataset.replace("DB", "DBP")
            else:
                dataset_arg = dataset
            if method in ["RDGCN", "BOOTEA"]:
                arg_map[method][dataset] = f"args_best/{method.lower()}_args_{dataset_arg.replace('-', '_')}.json"
            elif method == "TRANSEDGE":
                arg_map[method][dataset] = f"args_best/TransEdge_{dataset_arg.replace('-', '_')}.pkl"
            else:
                arg_map[method][dataset] = f"args_best/BERT_INT_{dataset_arg.replace('-', '_')}.pkl"
    for dataset in datasets:
        df_dataset = pd.DataFrame(
            columns=["Precision (Avg)", "Precision (Std)", "Recall (Avg)", "Recall (Std)", "F1 (Avg)", "F1 (Std)"],
            index=methods)
        df_dataset.index.name = "Method"
        for method in methods:
            if method in ["PARIS", "RDGCN", "BOOTEA", "TRANSEDGE"]:
                conda_command = 'conda run -n entity_match'
            else:
                conda_command = 'conda run -n bert-int'
            print(f"Running method: {method} for dataset: {dataset}")
            output_dir_log_fold_method = f"{output_dir_log_folds}/{method}"
            output_dir_log_exp_method_dataset = f"{output_dir_log_exp}/{method}_{dataset}.log"
            print(f"The final log will be written to: {output_dir_log_exp_method_dataset}")
            print(f"You can find fold specific logs into: {output_dir_log_fold_method}")
            if method != "PARIS":
                if method in ["RDGCN", "BOOTEA"]:
                    main_embeds = "../../OpenEA_Mod/run/main_from_args.py"
                elif method == "TRANSEDGE":
                    main_embeds = "../../TransEdge/code/transedge_ea.py"
                else:
                    main_embeds = "../../bert-int/run_full_bert.py"
                command = f"{conda_command} python3 -u ../run_experiment.py \
                            --method {method} \
                            --root_dataset {args.root_dataset}/OpenEA \
                            --dataset {dataset} \
                            --dataset_division 721_5fold \
                            --out_folder {output_dir_log_fold_method} \
                            --gpu {gpu} \
                            --main_embeds {main_embeds} \
                            --args {arg_map[method][dataset]} > {output_dir_log_exp_method_dataset}"
                os.system(command)
                with open(output_dir_log_exp_method_dataset) as f:
                    log_data = f.read()
                precision_no_csls_avg = float(
                    log_data.split("precisions no csls:\n\t")[1].split("avg: ")[1].split("\n\t")[0])
                precision_no_csls_std = float(
                    log_data.split("precisions no csls:\n\t")[1].split("avg: ")[1].split("\n\t")[1].split("std: ")[
                        1].split("\n")[0])
                recall_no_csls_avg = float(log_data.split("recalls no csls:\n\t")[1].split("avg: ")[1].split("\n\t")[0])
                recall_no_csls_std = float(
                    log_data.split("recalls no csls:\n\t")[1].split("avg: ")[1].split("\n\t")[1].split("std: ")[
                        1].split("\n")[0])
                f1_no_csls_avg = float(log_data.split("f1s no csls:\n\t")[1].split("avg: ")[1].split("\n\t")[0])
                f1_no_csls_std = float(
                    log_data.split("f1s no csls:\n\t")[1].split("avg: ")[1].split("\n\t")[1].split("std: ")[1].split(
                        "\n")[0])
                if method != "BERT-INT":
                    precision_csls_avg = float(
                        log_data.split("precisions csls:\n\t")[1].split("avg: ")[1].split("\n\t")[0])
                    precision_csls_std = float(
                        log_data.split("precisions csls:\n\t")[1].split("avg: ")[1].split("\n\t")[1].split("std: ")[
                            1].split("\n")[0])
                    recall_csls_avg = float(
                        log_data.split("recalls csls:\n\t")[1].split("avg: ")[1].split("\n\t")[0])
                    recall_csls_std = float(
                        log_data.split("recalls csls:\n\t")[1].split("avg: ")[1].split("\n\t")[1].split("std: ")[
                            1].split("\n")[0])
                    f1_csls_avg = float(log_data.split("f1s csls:\n\t")[1].split("avg: ")[1].split("\n\t")[0])
                    f1_csls_std = float(
                        log_data.split("f1s csls:\n\t")[1].split("avg: ")[1].split("\n\t")[1].split("std: ")[
                            1].split("\n")[0])
                precision_avg = precision_no_csls_avg if method == "BERT-INT" else max(precision_csls_avg,
                                                                                       precision_no_csls_avg)
                precision_std = precision_no_csls_std if (
                            method == "BERT-INT" or precision_avg == precision_no_csls_avg) else precision_csls_avg
                recall_avg = recall_no_csls_avg if method == "BERT-INT" else max(recall_csls_avg, recall_no_csls_avg)
                recall_std = recall_no_csls_std if (
                            method == "BERT-INT" or recall_avg == recall_no_csls_avg) else recall_csls_avg
                f1_avg = f1_no_csls_avg if method == "BERT-INT" else max(f1_csls_avg, f1_no_csls_avg)
                f1_std = f1_no_csls_std if (method == "BERT-INT" or f1_avg == f1_no_csls_avg) else f1_csls_avg
            else:
                command = f"{conda_command} python3 -u ../run_experiment.py \
                            --method PARIS \
                            --root_dataset {args.root_dataset}/OpenEA \
                            --dataset {dataset} \
                            --dataset_division 721_5fold \
                            --out_folder {output_dir_log_fold_method} > {output_dir_log_exp_method_dataset}"
                os.system(command)
                with open(output_dir_log_exp_method_dataset) as f:
                    log_data = f.read()
                precision_avg = float(log_data.split("precisions:\n\t")[1].split("avg: ")[1].split("\n\t")[0])
                precision_std = float(
                    log_data.split("precisions:\n\t")[1].split("avg: ")[1].split("\n\t")[1].split("std: ")[1].split(
                        "\n")[0])
                recall_avg = float(log_data.split("recalls:\n\t")[1].split("avg: ")[1].split("\n\t")[0])
                recall_std = float(
                    log_data.split("recalls:\n\t")[1].split("avg: ")[1].split("\n\t")[1].split("std: ")[1].split("\n")[
                        0])
                f1_avg = float(log_data.split("f1s:\n\t")[1].split("avg: ")[1].split("\n\t")[0])
                f1_std = float(
                    log_data.split("f1s:\n\t")[1].split("avg: ")[1].split("\n\t")[1].split("std: ")[1].split("\n")[0])

            df_dataset.loc[method] = [precision_avg, precision_std, recall_avg, recall_std, f1_avg, f1_std]
        df_dataset.to_csv(f"{output_dir}/{dataset}.csv")
        print(f"Written results for dataset: {dataset}")
