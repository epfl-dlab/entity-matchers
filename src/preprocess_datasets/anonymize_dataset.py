import argparse
import os
import random


def main(root_folder, dataset, fold_folder):
    new_dataset_folder = root_folder + "/" + dataset + "_ANON"
    command = "cp -r {} {}".format(root_folder + "/" + dataset, new_dataset_folder)
    os.system(command)
    ids = {}
    used = set()
    with open(new_dataset_folder + "/rel_triples_1") as f:
        for l in f:
            (e1, _, e2) = l.rstrip("\n").split("\t")
            if e1 not in ids:
                id_1 = random.randint(0, 10000000)
                while id_1 in used:
                    id_1 = random.randint(0, 10000000)
                ids[e1] = "L" + str(id_1)
                used.add(id_1)
            if e2 not in ids:
                id_2 = random.randint(0, 10000000)
                while id_2 in used:
                    id_2 = random.randint(0, 10000000)
                ids[e2] = "L" + str(id_2)
                used.add(id_2)
    used = set()
    with open(new_dataset_folder + "/rel_triples_2") as f:
        for l in f:
            (e1, _, e2) = l.rstrip("\n").split("\t")
            if e1 not in ids:
                id_1 = random.randint(0, 10000000)
                while id_1 in used:
                    id_1 = random.randint(0, 10000000)
                ids[e1] = "R" + str(id_1)
                used.add(id_1)
            if e2 not in ids:
                id_2 = random.randint(0, 10000000)
                while id_2 in used:
                    id_2 = random.randint(0, 10000000)
                ids[e2] = "R" + str(id_2)
                used.add(id_2)
    lines = []
    with open(new_dataset_folder + "/rel_triples_1") as f:
        for l in f:
            (e1, r, e2) = l.rstrip("\n").split("\t")
            lines.append("{}\t{}\t{}\n".format(ids[e1], r, ids[e2]))
    with open(new_dataset_folder + "/rel_triples_1", "w") as f:
        f.writelines(lines)
    lines = []
    with open(new_dataset_folder + "/rel_triples_2") as f:
        for l in f:
            (e1, r, e2) = l.rstrip("\n").split("\t")
            lines.append("{}\t{}\t{}\n".format(ids[e1], r, ids[e2]))
    with open(new_dataset_folder + "/rel_triples_2", "w") as f:
        f.writelines(lines)

    lines = []
    with open(new_dataset_folder + "/attr_triples_1") as f:
        for l in f:
            (e1, r, a) = l.rstrip("\n").split("\t")
            lines.append("{}\t{}\t{}\n".format(ids[e1], r, a))
    with open(new_dataset_folder + "/attr_triples_1", "w") as f:
        f.writelines(lines)
    lines = []
    with open(new_dataset_folder + "/attr_triples_2") as f:
        for l in f:
            (e1, r, a) = l.rstrip("\n").split("\t")
            lines.append("{}\t{}\t{}\n".format(ids[e1], r, a))
    with open(new_dataset_folder + "/attr_triples_2", "w") as f:
        f.writelines(lines)

    lines = []
    with open(new_dataset_folder + "/ent_links") as f:
        for l in f:
            (e1, e2) = l.rstrip("\n").split("\t")
            lines.append("{}\t{}\n".format(ids[e1], ids[e2]))
    with open(new_dataset_folder + "/ent_links", "w") as f:
        f.writelines(lines)

    if "increasing_seed" not in new_dataset_folder:
        for direct in os.listdir(new_dataset_folder + "/" + fold_folder):
            for file in os.listdir(new_dataset_folder + "/" + fold_folder + "/" + direct):
                lines = []
                with open(new_dataset_folder + "/" + fold_folder + "/" + direct + "/" + file) as f:
                    for l in f:
                        (e1, e2) = l.rstrip("\n").split("\t")
                        lines.append("{}\t{}\n".format(ids[e1], ids[e2]))
                with open(new_dataset_folder + "/" + fold_folder + "/" + direct + "/" + file, "w") as f:
                    f.writelines(lines)
    else:
        for direct in os.listdir(new_dataset_folder + "/" + fold_folder):
            for seed in os.listdir(new_dataset_folder + "/" + fold_folder + "/" + direct):
                for file in os.listdir(new_dataset_folder + "/" + fold_folder + "/" + direct + "/" + seed):
                    lines = []
                    with open(new_dataset_folder + "/" + fold_folder + "/" + direct + "/" + seed + "/" + file) as f:
                        for l in f:
                            (e1, e2) = l.rstrip("\n").split("\t")
                            lines.append("{}\t{}\n".format(ids[e1], ids[e2]))
                    with open(new_dataset_folder + "/" + fold_folder + "/" + direct + "/" + seed + "/" + file, "w") as f:
                        f.writelines(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Anonymize the given dataset")

    parser.add_argument(
        "--root_folder",
        type=str,
        help='Name of the root directory (it will be used even as output root.)'
    )

    parser.add_argument(
        "--dataset",
        help="Name of the file containing the dataset",
        required=True
    )

    parser.add_argument(
        "--fold_folder",
        help="Fold folder to anonymize",
        required=True
    )

    args = parser.parse_args()

    main(args.root_folder, args.dataset, args.fold_folder)
