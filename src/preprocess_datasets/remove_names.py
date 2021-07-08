import argparse
import os


def main(root_folder, dataset, out_root):
    new_dataset_folder = out_root + "/" + dataset.replace("_NO_EXTRA", "") + "_NO_NAMES"
    command = "cp -r {} {}".format(root_folder + "/" + dataset, new_dataset_folder)
    os.system(command)
    names_to_remove = {
        "DBP_en_WD_en_15K_NO_EXTRA": {"http://xmlns.com/foaf/0.1/name", "http://www.wikidata.org/prop/direct/P373",
                                      "http://www.w3.org/2004/02/skos/core#altLabel",
                                      "http://www.wikidata.org/prop/direct/P935"},
        "DBP_en_YG_en_15K_NO_EXTRA": {"http://xmlns.com/foaf/0.1/name", "http://dbpedia.org/ontology/birthName",
                                      "redirectedFrom", "skos:prefLabel"}
    }
    lines = []
    with open(new_dataset_folder + "/attr_triples_1") as f:
        for l in f:
            (e, r, a) = l.rstrip("\n").split("\t")
            if r not in names_to_remove[dataset]:
                lines.append("{}\t{}\t{}\n".format(e, r, a))
    with open(new_dataset_folder + "/attr_triples_1", "w") as f:
        f.writelines(lines)
    lines = []
    with open(new_dataset_folder + "/attr_triples_2") as f:
        for l in f:
            (e, r, a) = l.rstrip("\n").split("\t")
            if r not in names_to_remove[dataset]:
                lines.append("{}\t{}\t{}\n".format(e, r, a))
    with open(new_dataset_folder + "/attr_triples_2", "w") as f:
        f.writelines(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Remove names from the dataset")

    parser.add_argument(
        "--root_folder",
        type=str,
        help='Name of the root directory'
    )

    parser.add_argument(
        "--dataset",
        help="Name of the file containing the dataset",
        required=True
    )

    parser.add_argument(
        "--out_root",
        help="Path to output root",
        required=True
    )

    args = parser.parse_args()

    main(args.root_folder, args.dataset, args.out_root)
