from collections import defaultdict
from tqdm import tqdm
import argparse

def check_sample(KG1_rel_triples, KG2_rel_triples, full_truth, sample_truth):
    kg1_truth = defaultdict(str)
    kg2_truth = defaultdict(str)
    out_truth = full_truth.difference(sample_truth)
    ents_kg1 = set([e for (e, _, _) in KG1_rel_triples]) | set([e for (_, _, e) in KG1_rel_triples])
    ents_kg2 = set([e for (e, _, _) in KG2_rel_triples]) | set([e for (_, _, e) in KG2_rel_triples])
    for (e1, e2) in tqdm(out_truth):
        if e1 in ents_kg1:
            kg1_truth[e1] = e2
        if e2 in ents_kg2:
            kg2_truth[e2] = e1
    for e1 in tqdm(ents_kg1):
        if kg1_truth[e1] in ents_kg2:
            print(e1, kg1_truth[e1])
            print("ERROR")
            return
    for e2 in tqdm(ents_kg2):
        if kg2_truth[e2] in ents_kg1:
            print(e2, kg2_truth[e2])
            print("ERROR")
            return
    print("OK")


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Check obtained sample')
    parser.add_argument('--root_folder', type=str, help="Target dataset folder")
    args = parser.parse_args()
    rel_triples1 = set()
    with open("{}/rel_triples_1".format(args.root_folder)) as f:
        for l in tqdm(f):
            (e1, p, e2) = l.rstrip("\n").rstrip().split("\t")
            rel_triples1.add((e1, p, e2))
    rel_triples2 = set()
    with open("{}/rel_triples_2".format(args.root_folder)) as f:
        for l in tqdm(f):
            (e1, p, e2) = l.rstrip("\n").rstrip().split("\t")
            rel_triples2.add((e1, p, e2))
    full_truth = set()
    with open("{}/ent_links_full".format(args.root_folder)) as f:
        for l in tqdm(f):
            (e1, e2) = l.rstrip("\n").rstrip().split("\t")
            full_truth.add((e1, e2))
    sample_truth = set()
    with open("{}/ent_links".format(args.root_folder)) as f:
        for l in tqdm(f):
            (e1, e2) = l.rstrip("\n").rstrip().split("\t")
            sample_truth.add((e1, e2))
    check_sample(rel_triples1, rel_triples2, full_truth, sample_truth)