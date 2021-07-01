import pandas as pd
import numpy as np
import os, shutil
import argparse
import data
from performance import check_result
from args import load_args


def main(args):
    """
    Main function to run experiment on PARIS
    """
    if args.fold_number == "full":
        valid_list = []
        test_list = []
        truth_list = []
        for fold in range(1,6):
            args.fold_number = str(fold)
            triples1, triples2, folder = data.load_kgs(args.training_data, args.dataset_type, args.triples_type, args.dataset_ea)
            if args.seed:
                kg1_path, kg2_path = data.seed_kgs(args.dataset_type, triples1, triples2, folder, args.new_seed, args.dataset_division, args.fold_number, args.frac)
            else:
                kg1_path, kg2_path = data.add_point(triples1, triples2, folder)
            
            # Delete old PARIS directory and create again to be empty
            out_path = args.output + "PARIS/" + args.dataset_type + "/" + args.dataset_ea
            if os.path.exists(out_path):
                shutil.rmtree(out_path)
            os.mkdir(out_path)

            # Run PARIS
            print("Running PARIS...")
            out_paris = os.popen(
                "java -jar ../resources/paris_0_3.jar {kg1_path} {kg2_path} {out_path}".format(
                    kg1_path=kg1_path, kg2_path=kg2_path, out_path=out_path
                )
            ).read()
            print("PARIS output:\n", out_paris)

            print("\nComputing metrics")
            if args.dataset_type == "OpenEA_dataset":
                prec_truth, rec_truth, f1_truth, prec_test, rec_test, f1_test, prec_valid, rec_valid, f1_valid = \
                    check_result(folder, args.dataset_type, out_path, args.dataset_division, args.fold_number)
                truth_list.append((prec_truth, rec_truth, f1_truth))
                test_list.append((prec_test, rec_test, f1_test))
                valid_list.append((prec_valid, rec_valid, f1_valid))
            else:
                check_result(folder, args.dataset_type, out_path)
            
            print("\nCleaning")
            if args.seed and not args.new_seed:
                data.clean(folder, args.dataset_type, out_path, args.dataset_division, args.fold_number)
            else:
                data.clean(folder, args.dataset_type, out_path)
        
        print("Truth average precision:", np.mean(np.array([t[0] for t in truth_list])))
        print("Truth std precision:", np.std(np.array([t[0] for t in truth_list])))
        print("Truth average recall:", np.mean(np.array([t[1] for t in truth_list])))
        print("Truth std recall:", np.std(np.array([t[1] for t in truth_list])))
        print("Truth average f1:", np.mean(np.array([t[2] for t in truth_list])))        
        print("Truth std f1:", np.std(np.array([t[2] for t in truth_list])))
        print("Test average precision:", np.mean(np.array([t[0] for t in test_list])))
        print("Test std precision:", np.std(np.array([t[0] for t in test_list])))
        print("Test average recall:", np.mean(np.array([t[1] for t in test_list])))
        print("Test std recall:", np.std(np.array([t[1] for t in test_list])))
        print("Test average f1:", np.mean(np.array([t[2] for t in test_list])))
        print("Test std f1:", np.std(np.array([t[2] for t in test_list])))
        print("Valid average precision:", np.mean(np.array([t[0] for t in valid_list])))
        print("Valid std precision:", np.std(np.array([t[0] for t in valid_list])))
        print("Valid average recall:", np.mean(np.array([t[1] for t in valid_list])))
        print("Valid std recall:", np.std(np.array([t[1] for t in valid_list])))
        print("Valid average f1:", np.mean(np.array([t[2] for t in valid_list])))
        print("Valid std f1:", np.std(np.array([t[2] for t in valid_list])))

    else:
        triples1, triples2, folder = data.load_kgs(args.training_data, args.dataset_type, args.triples_type, args.dataset_ea)
        if args.seed:
            kg1_path, kg2_path = data.seed_kgs(args.dataset_type, triples1, triples2, folder, args.new_seed, args.dataset_division, args.fold_number, args.frac)
        else:
            kg1_path, kg2_path = data.add_point(triples1, triples2, folder)
        
        # Delete old PARIS directory and create again to be empty
        out_path = args.output + "PARIS/" + args.dataset_type + "/" + args.dataset_ea
        if os.path.exists(out_path):
            shutil.rmtree(out_path)
        os.mkdir(out_path)

        # Run PARIS
        print("Running PARIS...")
        out_paris = os.popen(
            "java -jar ../resources/paris_0_3.jar {kg1_path} {kg2_path} {out_path}".format(
                kg1_path=kg1_path, kg2_path=kg2_path, out_path=out_path
            )
        ).read()
        print("PARIS output:\n", out_paris)

        print("\nComputing metrics")
        if args.dataset_type == "OpenEA_dataset":
            if args.seed:
                check_result(folder, args.dataset_type, out_path, args.dataset_division, args.fold_number)
            else:
                check_result(folder, args.dataset_type, out_path)
        else:
            check_result(folder, args.dataset_type, out_path)
        
        print("\nCleaning")
        if args.seed and not args.new_seed:
            data.clean(folder, args.dataset_type, out_path, args.dataset_division, args.fold_number)
        else:
            data.clean(folder, args.dataset_type, out_path)
    print("\nNothing else to do! Closing")


if __name__ == "__main__":
    """
    Run main with argument
    """
    parser = argparse.ArgumentParser(
        description="Run PARIS entity matching and compute metrics"
    )

    parser.add_argument(
        "--json_args"
    )
    parser.add_argument(
        "--dataset_ea",
        default = ""
    )
    parser.add_argument(
        "--fold_number",
        default = None
    )

    args_main = parser.parse_args()
    args = load_args(args_main.json_args)
    args.dataset_ea = args_main.dataset_ea
    args.fold_number = args_main.fold_number
    main(args)
