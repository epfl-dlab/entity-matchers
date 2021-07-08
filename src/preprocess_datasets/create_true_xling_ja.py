import argparse
import os
import numpy as np

ranges = [
    {"from": ord(u"\u3300"), "to": ord(u"\u33ff")},  # compatibility ideographs
    {"from": ord(u"\ufe30"), "to": ord(u"\ufe4f")},  # compatibility ideographs
    {"from": ord(u"\uf900"), "to": ord(u"\ufaff")},  # compatibility ideographs
    {"from": ord(u"\U0002F800"), "to": ord(u"\U0002fa1f")},  # compatibility ideographs
    {'from': ord(u'\u3040'), 'to': ord(u'\u309f')},  # Japanese Hiragana
    {"from": ord(u"\u30a0"), "to": ord(u"\u30ff")},  # Japanese Katakana
    {"from": ord(u"\u2e80"), "to": ord(u"\u2eff")},  # cjk radicals supplement
    {"from": ord(u"\u4e00"), "to": ord(u"\u9fff")},
    {"from": ord(u"\u3400"), "to": ord(u"\u4dbf")},
    {"from": ord(u"\U00020000"), "to": ord(u"\U0002a6df")},
    {"from": ord(u"\U0002a700"), "to": ord(u"\U0002b73f")},
    {"from": ord(u"\U0002b740"), "to": ord(u"\U0002b81f")},
    {"from": ord(u"\U0002b820"), "to": ord(u"\U0002ceaf")}  # included as of Unicode 8.0
]


def is_cjk(char):
    return any([range["from"] <= ord(char) <= range["to"] for range in ranges])


def has_jap(string):
    return np.array([is_cjk(c) for c in string]).sum() > 0


def main(dataset_folder):
    new_dataset_folder = dataset_folder + "_TRUE_XLING"
    command = "cp -r {} {}".format(dataset_folder, new_dataset_folder)
    os.system(command)
    lines = []
    with open(new_dataset_folder + "/attr_triples_1") as f:
        for l in f:
            (e, r, a) = l.rstrip("\n").split("\t")
            if not has_jap(a):
                lines.append("{}\t{}\t{}\n".format(e, r, a))
    with open(new_dataset_folder + "/attr_triples_1", "w") as f:
        f.writelines(lines)
    lines = []
    with open(new_dataset_folder + "/attr_triples_2") as f:
        for l in f:
            (e, r, a) = l.rstrip("\n").split("\t")
            if has_jap(a):
                lines.append("{}\t{}\t{}\n".format(e, r, a))
    with open(new_dataset_folder + "/attr_triples_2", "w") as f:
        f.writelines(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create true xlingual japanese dataset")

    parser.add_argument(
        "--dataset_folder",
        help="Name of the file containing the dataset",
        required=True
    )

    args = parser.parse_args()

    main(args.dataset_folder)
