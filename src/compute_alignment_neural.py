import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial.distance import cdist
from scipy.special import softmax
import pickle
import time
import torch
import torch.nn as nn

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Check KG statistics')
    parser.add_argument('--in_results_folder', type=str,
                        help="Folder where the embedding methods put the embeddings/entity ids like 20201112...")
    parser.add_argument('--in_links', type=str, help="Folder where train/test/valid links are stored")
    parser.add_argument('--method', type=str, choices=['bootea', 'rdgcn'], help="Which method to use (bootea, rdgcn)")
    parser.add_argument('--temperatures', nargs='+', type=float, help='Temperatures to try')
    parser.add_argument('--thresholds', nargs='+', type=float, help='Thresholds to try')
    parser.add_argument('--gpu', type=str, help="Which GPU to use")
    args = parser.parse_args()

    similarity_function = {}
    similarity_function['bootea'] = lambda x1, x2: torch.mv(x2, x1).reshape(-1, 1).T
    similarity_function['rdgcn'] = \
        lambda x1, x2: 1 - torch.cdist(x1.reshape(1, -1), x2, p=1.0)

    if torch.cuda.is_available():
        device = "cuda:{}".format(args.gpu)
    else:
        device = "cpu"

    ent_embeds = np.load("{}/ent_embeds.npy".format(args.in_results_folder))
    # name_entity -> original_id
    kg1_id = {}
    kg2_id = {}
    # original_id -> name_entity
    kg1_id_to_ent = {}
    kg2_id_to_ent = {}
    with open("{}/kg1_ent_ids".format(args.in_results_folder)) as f:
        for line in f:
            (ent, idx) = line.rstrip("\n").split("\t")
            kg1_id[ent] = int(idx)
            kg1_id_to_ent[int(idx)] = ent
    with open("{}/kg2_ent_ids".format(args.in_results_folder)) as f:
        for line in f:
            (ent, idx) = line.rstrip("\n").split("\t")
            kg2_id[ent] = int(idx)
            kg2_id_to_ent[int(idx)] = ent

    # (entity_name_kg1, entity_name_kg2)
    train_links = set()
    test_links = set()
    valid_links = set()
    # (original_id_kg1, original_id_kg2)
    train_links_id = set()
    test_links_id = set()
    valid_links_id = set()
    with open("{}/test_links".format(args.in_links)) as f:
        for line in f:
            (e1, e2) = line.rstrip("\n").split("\t")
            test_links.add((e1, e2))
            test_links_id.add((kg1_id[e1], kg2_id[e2]))

    with open("{}/train_links".format(args.in_links)) as f:
        for line in f:
            (e1, e2) = line.rstrip("\n").split("\t")
            train_links.add((e1, e2))
            train_links_id.add((kg1_id[e1], kg2_id[e2]))

    with open("{}/valid_links".format(args.in_links)) as f:
        for line in f:
            (e1, e2) = line.rstrip("\n").split("\t")
            valid_links.add((e1, e2))
            valid_links_id.add((kg1_id[e1], kg2_id[e2]))

    # List of ids not contained in the training data.
    # pos -> original_id_kg1
    id1_map_new_to_old = list(set(kg1_id.values()).difference(set([id1 for (id1, id2) in train_links_id])))
    id2_map_new_to_old = list(set(kg2_id.values()).difference(set([id2 for (id1, id2) in train_links_id])))
    # Embedding now are indexed with the id1_map_new_to_old, starting from 0
    ent_embeds1 = torch.from_numpy(ent_embeds[id1_map_new_to_old]).float().to(device)
    ent_embeds2 = torch.from_numpy(ent_embeds[id2_map_new_to_old]).float().to(device)
    aligns = {}
    start_time = time.time()
    soft_max = nn.Softmax(dim=1)
    for temperature in args.temperatures:
        aligns[temperature] = {}
        # Used to store which entities have not been matched yet
        masks_map = {}
        for thresh in args.thresholds:
            masks_map[thresh] = torch.ones(ent_embeds2.shape[0], device=device) == 1
            aligns[temperature][thresh] = []
        for i in tqdm(range(ent_embeds1.shape[0])):
            similarities = similarity_function[args.method](ent_embeds1[i], ent_embeds2)
            similarities = soft_max(temperature * similarities).reshape(-1, 1)
            for threshold in args.thresholds:
                max_sim = torch.max(similarities[masks_map[threshold].reshape(-1, 1)]).item()
                if max_sim < threshold:
                    aligns[temperature][threshold].append((i,))
                    continue
                bool_sim = (similarities == max_sim).reshape(-1) & masks_map[threshold]
                idx_max = torch.where(bool_sim == True)[0][0]
                masks_map[threshold][idx_max.item()] = False
                aligns[temperature][threshold].append((i, idx_max.item()))
    print("Total time elapsed: " + str(time.time() - start_time))
    aligns_original = {}
    for temperature in args.temperatures:
        aligns_original[temperature] = {}
        for threshold in args.thresholds:
            aligns_original[temperature][threshold] = []
    for temperature in args.temperatures:
        for threshold in args.thresholds:
            for align in aligns[temperature][threshold]:
                if len(align) == 2:
                    id1, id2 = align
                    aligns_original[temperature][threshold].append(
                        (kg1_id_to_ent[id1_map_new_to_old[id1]], kg2_id_to_ent[id2_map_new_to_old[id2]]))
                else:
                    aligns_original[temperature][threshold].append(kg1_id_to_ent[id1_map_new_to_old[align[0]]])
    with open("{}/greedy_alignments.pkl".format(args.in_results_folder), "wb") as f:
        pickle.dump(aligns_original, f)
    print("Results saved!")
