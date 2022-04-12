"""
This script will reproduce the results in Figure 2 of the paper. The figure will be stored in ./output/output_figure_2.
You can also find the detailed logs for each run and experiment into
./output/output_figure_2/log_folds and ./output/output_figure_2/log_exp.

This script assume that you created the environments to run all the experiments as specified in our README.md.
You need to have one environment entity_match for all the methods but BERT-INT and one called bert-int for BERT-INT

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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def create_figure(df_dict, methods, output_dir):
    markers = {"BOOTEA": "o", "TRANSEDGE": "v", "RDGCN": "^", "BERT-INT": "s", "PARIS": "D", "Ditto": "X"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

    df_d_w = df_dict["DB-WD-100K"]
    df_d_y = df_dict["DB-YG-100K"]

    for method in methods:
        ax1.plot(df_d_w.loc[method].index, df_d_w.loc[method]["F1"], label=method, marker=markers[method], linewidth=5,
                 markersize=20, fillstyle="none")
        ax2.plot(df_d_y.loc[method].index, df_d_y.loc[method]["F1"], label=method, marker=markers[method], linewidth=5,
                 markersize=20, fillstyle="none")

    ax1.set_xticks(df_d_w.loc["BOOTEA"].index.to_numpy())
    ax2.set_xticks(df_d_y.loc["BOOTEA"].index.to_numpy())
    ax1.set_title("(a) DB-WD-100K", fontsize=24, y=-0.27)
    ax2.set_title("(b) DB-YG-100K", fontsize=24, y=-0.27)
    ax1.set_ylabel("F1", fontsize=22)
    ax1.set_xlabel("Seed%", fontsize=22)
    ax2.set_xlabel("Seed%", fontsize=22)
    ax1.tick_params(axis='both', which='major', labelsize=20)
    ax2.tick_params(axis='both', which='major', labelsize=20)
    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=18, frameon=False, ncol=len(labels), loc='upper center')
    ax1.grid()
    ax2.grid()
    fig.savefig(f"{output_dir}/figure2.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproduce Figure 2.")
    parser.add_argument("--root_dataset", type=str,
                        help="Path to the root folder containing the downloaded datasets (no slash)")
    parser.add_argument("--gpu", type=str, help="GPU id to use (if present)")
    args = parser.parse_args()

    methods = ["BOOTEA", "TRANSEDGE", "RDGCN", "BERT-INT", "PARIS"]
    datasets = ["DB-WD-100K", "DB-YG-100K"]
    seeds = [1, 25, 50, 75, 89]
    output_base = "./output"
    output_dir = f"{output_base}/output_figure_2"
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
    dict_df = {}
    for dataset in datasets:
        index = pd.MultiIndex.from_product([methods, seeds],
                                           names=['Method', 'Seed%'])
        df_dataset = pd.DataFrame(columns=["F1"], index=index)
        for seed in seeds:
            for method in methods:
                if method in ["PARIS", "RDGCN", "BOOTEA", "TRANSEDGE"]:
                    conda_command = 'conda run -n entity_match'
                else:
                    conda_command = 'conda run -n bert-int'
                print(f"Running method: {method} for dataset: {dataset} and seed: {seed}%")
                output_dir_log_fold_method = f"{output_dir_log_folds}/{method}"
                output_dir_log_exp_method_dataset_seed = f"{output_dir_log_exp}/{method}_{dataset}_{seed}_seed.log"
                print(f"The final log will be written to: {output_dir_log_exp_method_dataset_seed}")
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
                                --root_dataset {args.root_dataset}/SupRealEA_New \
                                --dataset {dataset} \
                                --dataset_division 0_18_91_1folds/{seed}_seed \
                                --out_folder {output_dir_log_fold_method} \
                                --gpu {gpu} \
                                --main_embeds {main_embeds} \
                                --args {arg_map[method][dataset]} > {output_dir_log_exp_method_dataset_seed}"
                    os.system(command)
                    with open(output_dir_log_exp_method_dataset_seed) as f:
                        log_data = f.read()
                    try:
                        f1_no_csls_avg = float(log_data.split("f1s no csls:\n\t")[1].split("avg: ")[1].split("\n\t")[0])
                        if method != "BERT-INT":
                            f1_csls_avg = float(log_data.split("f1s csls:\n\t")[1].split("avg: ")[1].split("\n\t")[0])
                        f1_avg = f1_no_csls_avg if method == "BERT-INT" else max(f1_csls_avg, f1_no_csls_avg)
                    except Exception:
                        print("Error while executing the current method, F1 = NaN")
                        f1_avg = np.NaN
                else:
                    command = f"{conda_command} python3 -u ../run_experiment.py \
                                --method PARIS \
                                --root_dataset {args.root_dataset}/SupRealEA_New \
                                --dataset {dataset} \
                                --dataset_division 0_18_91_1folds/{seed}_seed \
                                --out_folder {output_dir_log_fold_method} > {output_dir_log_exp_method_dataset_seed}"
                    os.system(command)
                    with open(output_dir_log_exp_method_dataset_seed) as f:
                        log_data = f.read()
                    f1_avg = float(log_data.split("f1s:\n\t")[1].split("avg: ")[1].split("\n\t")[0])
                df_dataset.loc[method, seed] = [f1_avg]
        dict_df[dataset] = df_dataset
    print("Finished running the methods, creating the figure!")
    create_figure(dict_df, methods, output_dir)
    print(f"Figure saved to {output_dir}/figure2.png")
