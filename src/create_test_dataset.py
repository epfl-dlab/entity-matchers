import argparse
import os
from tqdm import tqdm
import random

parser = argparse.ArgumentParser(
    description="Create smaller tests datasets to conduct tests on the sampling procedure. "
)

parser.add_argument(
    "--dataset1",
    type=str,
    help='First dataset name'
)

parser.add_argument(
    "--dataset2",
    type=str,
    help='Second dataset name'
)

parser.add_argument(
    "--alignment",
    type=str,
    help='Full path to the alignment file. Assume always first column is kg1 and second column kg2'
)

parser.add_argument(
    "--root_folder_kg1",
    type=str,
    help='Name of the root directory of the dataset (attr_triples and rel_triples automatically loaded)'
)

parser.add_argument(
    "--root_folder_kg2",
    type=str,
    help='Name of the root directory of the dataset (attr_triples and rel_triples automatically loaded)'
)

parser.add_argument(
    "--out_folder",
    type=str,
    help='Output folder name'
)

parser.add_argument(
    "--truth_size",
    type=int,
    help='How many entities keep randomly from the truth (non-truth entities will be randomly chosen'
         ' to keep the same ratio as the original datasets)'
)


def truth_pairs(align_path, ents_1, ents_2):
    pairs_list = []
    print("Reading entities from the truth...")
    with open(align_path, "r") as f_pairs:
        for l in tqdm(f_pairs):
            (el, er) = l.rstrip("\n").split("\t")
            if el in ents_1 and er in ents_2:
                pairs_list.append((el, er))
    return pairs_list


def read_ents_and_triples(triples_file, rel_ents=None):
    if rel_ents is None:
        rel_ents = set()
    attr_ents = set()
    triples = []
    print("Reading " + triples_file + "...")
    with open(triples_file, "r") as f_triples:
        for l in tqdm(f_triples):
            (h, r, t) = l.rstrip("\n").split("\t")
            triples.append((h, r, t))
            if "rel_triples" in triples_file:
                rel_ents.add(h)
                rel_ents.add(t)
            else:
                attr_ents.add(h)
    if "rel_triples" in triples_file:
        return rel_ents, triples
    else:
        return rel_ents.intersection(attr_ents), triples


def split_truth(kg_ent, ent_pairs, idx):
    in_truth = set()
    out_truth = set()
    truth_ent = set([pair[idx] for pair in ent_pairs])
    for ent in tqdm(kg_ent):
        if ent in truth_ent:
            in_truth.add(ent)
        else:
            out_truth.add(ent)
    return in_truth, out_truth


def write_out(kg, out, rel_triples, attr_triples, sample_truth, selected_out):

    final_ents = set()
    print("Saving filtered relations for kg " + kg + "...")
    new_rel = []
    for (h, r, t) in tqdm(rel_triples):
        if h in sample_truth or t in sample_truth or h in selected_out or t in selected_out:
            new_rel.append("\t".join((h, r, t)) + "\n")
            final_ents.add(h)
            final_ents.add(t)
    with open("{}/rel_triples_{}".format(out, kg), "w") as f_rel:
        f_rel.writelines(new_rel)

    print("Saving filtered attributes for kg " + kg + "...")
    new_attr = []
    for (h, r, t) in tqdm(attr_triples):
        if h in sample_truth or h in selected_out:
            new_attr.append("\t".join((h, r, t)) + "\n")
            final_ents.add(h)
    with open("{}/attr_triples_{}".format(out, kg), "w") as f_attr:
        f_attr.writelines(new_attr)

    print("Total entities in the kg " + kg + ": " + str(len(final_ents)))

if __name__ == "__main__":
    args_main = parser.parse_args()

    dataset1 = args_main.dataset1
    dataset2 = args_main.dataset2
    alignment = args_main.alignment
    root_folder_kg1 = args_main.root_folder_kg1
    root_folder_kg2 = args_main.root_folder_kg2
    out_folder = args_main.out_folder
    truth_size = args_main.truth_size

    kg1_ent, rel_triples1 = read_ents_and_triples("{}/rel_triples".format(root_folder_kg1))
    kg1_ent, attr_triples1 = read_ents_and_triples("{}/attr_triples".format(root_folder_kg1), kg1_ent)
    print("Total entities in both attributes and relations kg1: " + str(len(kg1_ent)))
    print("Relation triples kg1: " + str(len(rel_triples1)))
    print("Attribute triples kg1: " + str(len(attr_triples1)))
    kg2_ent, rel_triples2 = read_ents_and_triples("{}/rel_triples".format(root_folder_kg2))
    kg2_ent, attr_triples2 = read_ents_and_triples("{}/attr_triples".format(root_folder_kg2), kg2_ent)
    print("Total entities in both attributes and relations kg2: " + str(len(kg2_ent)))
    print("Relation triples kg2: " + str(len(rel_triples2)))
    print("Attribute triples kg2: " + str(len(attr_triples2)))
    pairs = truth_pairs(alignment, kg1_ent, kg2_ent)
    print("Total entities truth (present in both KGs): " + str(len(pairs)))

    in_truth1, out_truth1 = split_truth(kg1_ent, pairs, 0)
    print("Entities inside the truth for KG1: " + str(len(in_truth1)))
    print("Entities not in the truth for KG1: " + str(len(out_truth1)))
    ratio1 = len(out_truth1) / len(in_truth1)
    print("Ratio 1: " + str(ratio1))
    in_truth2, out_truth2 = split_truth(kg2_ent, pairs, 1)
    print("Entities inside the truth for KG2: " + str(len(in_truth2)))
    print("Entities not in the truth for KG2: " + str(len(out_truth2)))
    ratio2 = len(out_truth2) / len(in_truth2)
    print("Ratio 2: " + str(ratio2))

    selected_truth = random.sample(pairs, truth_size)
    selected_out1 = random.sample(out_truth1, int(truth_size * ratio1))
    selected_out1_dict = {}
    for e in selected_out1:
        selected_out1_dict[e] = True
    selected_out2 = random.sample(out_truth2, int(truth_size * ratio2))
    selected_out2_dict = {}
    for e in selected_out2:
        selected_out2_dict[e] = True
    print("Random truth size: " + str(len(selected_truth)))
    print("Random out size KG1: " + str(len(selected_out1)))
    print("Random out size KG2: " + str(len(selected_out2)))

    out_full = "{}/{}-{}_{}".format(out_folder, dataset1, dataset2, truth_size)
    if not os.path.exists(out_full):
        os.makedirs(out_full)

    selected_truth1 = set([pair[0] for pair in selected_truth])
    sel_truth_dict1 = {}
    for e in selected_truth1:
        sel_truth_dict1[e] = True
    write_out(dataset1, out_full, rel_triples1, attr_triples1, sel_truth_dict1, selected_out1_dict)
    selected_truth2 = set([pair[1] for pair in selected_truth])
    sel_truth_dict2 = {}
    for e in selected_truth2:
        sel_truth_dict2[e] = True
    write_out(dataset2, out_full, rel_triples2, attr_triples2, sel_truth_dict2, selected_out2_dict)

    print("Saving filtered links...")
    new_pairs = []
    for (e1, e2) in tqdm(selected_truth):
        new_pairs.append("\t".join((e1, e2)) + "\n")
    with open(out_full + "/" + "ent_links", "w") as f:
        f.writelines(new_pairs)
    print("End!")