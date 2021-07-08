import argparse
from tqdm import tqdm
import pickle

def build_map_rel(non_anon_dataset_folder, anon_dataset_folder, map_no_to_anon, map_anon_to_no, rel_file_num):
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert description pkl to an anonymous one.")

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

    parser.add_argument(
        "--original_desc",
        type=str,
        help="Path to the original description pkl",
        required=True
    )

    parser.add_argument(
        "--dataset",
        type=str,
        help="Name of the dataset (to be appended to description name)",
        required=True
    )

    args = parser.parse_args()
    map_no_to_anon = {}
    map_anon_to_no = {}
    build_map_rel(args.dataset_non_anon_folder, args.dataset_anon_folder, map_no_to_anon, map_anon_to_no, "1")
    build_map_rel(args.dataset_non_anon_folder, args.dataset_anon_folder, map_no_to_anon, map_anon_to_no, "2")
    new_pickle_path = "/".join(args.original_desc.split("/")[0:-1]) + "/desc_" + args.dataset + ".pkl"
    with open(args.original_desc, "rb") as f:
        original_desc = pickle.load(f)
    new_desc = {}
    for (e, d) in original_desc.items():
        if e not in map_no_to_anon:
            continue
        new_desc[map_no_to_anon[e]] = d
    with open(new_pickle_path, "wb") as f:
        pickle.dump(new_desc, f)