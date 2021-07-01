import sys
import random
import os

def change_links(links_path, map_ent_to_new):
    new_links = []
    with open(links_path) as f:
        for l in f:
            (e1, e2) = l.rstrip("\n").split("\t")
            new_links.append((e1, "Q" + str(map_ent_to_new[e2])))
    new_links = map(lambda x: x[0] + "\t" + x[1] + "\n", new_links)
    with open(links_path, "w") as f:
        f.writelines(new_links)

path = sys.argv[1]
fold_division = sys.argv[2]
map_ent_to_new = {}
used_ids = set()
new_rels = []
with open(path + "/rel_triples_2") as f:
    for l in f:
        (e1, r, e2) = l.rstrip("\n").split("\t")
        if e1 in map_ent_to_new:
            new_id1 = map_ent_to_new[e1]
        else:
            new_id1 = random.randint(0, 5000000)
            while new_id1 in used_ids:
                new_id1 = random.randint(0, 5000000)
            used_ids.add(new_id1)
            map_ent_to_new[e1] = new_id1
        if e2 in map_ent_to_new:
            new_id2 = map_ent_to_new[e2]
        else:
            new_id2 = random.randint(0, 5000000)
            while new_id2 in used_ids:
                new_id2 = random.randint(0, 5000000)
            used_ids.add(new_id2)
            map_ent_to_new[e2] = new_id2
        new_rels.append(("Q" + str(new_id1), r, "Q" + str(new_id2)))

new_rels = map(lambda x: x[0] + "\t" + x[1] + "\t" + x[2] + "\n", new_rels)
with open(path + "/rel_triples_2", "w") as f:
    f.writelines(new_rels)

change_links(path + "/ent_links", map_ent_to_new)
for fold in os.listdir(path + "/" + fold_division):
    fold_path = path + "/" + fold_division + "/" + fold
    change_links(fold_path + "/train_links", map_ent_to_new)
    change_links(fold_path + "/test_links", map_ent_to_new)
    change_links(fold_path + "/valid_links", map_ent_to_new)
