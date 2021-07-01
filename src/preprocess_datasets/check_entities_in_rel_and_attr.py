import sys
path = sys.argv[1]
ents_rel_1 = set()
ents_rel_2 = set()
ents_attr_1 = set()
ents_attr_2 = set()
print("Reading rel triples 1")
with open(path+"/rel_triples_1") as f:
    for l in f:
        (e1, r, e2) = l.rstrip("\n").split("\t")
        ents_rel_1.add(e1)
        ents_rel_1.add(e2)
print("Reading rel triples 2")
with open(path+"/rel_triples_2") as f:
    for l in f:
        (e1, r, e2) = l.rstrip("\n").split("\t")
        ents_rel_2.add(e1)
        ents_rel_2.add(e2)
print("Reading attr triples 1")
with open(path+"/attr_triples_1") as f:
    for l in f:
        (e1, p, o) = l.rstrip("\n").split("\t")
        ents_attr_1.add(e1)
print("Reading attr triples 2")
with open(path+"/attr_triples_2") as f:
    for l in f:
        try:
            (e1, p, o) = l.rstrip("\n").split("\t")
        except ValueError:
            continue
        ents_attr_2.add(e1)
print("Tot entities rel triples 1:", len(ents_rel_1))
print("Tot entities attr triples 1:", len(ents_attr_1))
print("Entities in rel triples 1 not in attr triples 1:", len(ents_rel_1 - ents_attr_1))
print("Entities in attr triples 1 not in rel triples 1:", len(ents_attr_1 - ents_rel_1))
print("Tot entities rel triples 2:", len(ents_rel_2))
print("Tot entities attr triples 2:", len(ents_attr_2))
print("Entities in rel triples 2 not in attr triples 2:", len(ents_rel_2 - ents_attr_2))
print("Entities in attr triples 2 not in rel triples 2:", len(ents_attr_2 - ents_rel_2))

print("Filtering seed")
new_seed = []
with open(path+"/ent_links") as f:
    for l in f:
        (e1, e2) = l.rstrip("\n").split("\t")
        if e1 in ents_rel_1 and e2 in ents_rel_2:
            new_seed.append((e1, e2))
new_seed = map(lambda x: x[0] + "\t" + x[1] + "\n", new_seed)
with open(path+"/ent_links_new", "w") as f:
    f.writelines(new_seed)

new_attr_1 = []
new_attr_2 = []
print("Filtering attr 1")
with open(path+"/attr_triples_1") as f:
    for l in f:
        (e1, p, o) = l.rstrip("\n").split("\t")
        if e1 in ents_rel_1:
            new_attr_1.append((e1, p, o))
print("Filtering attr 2")
with open(path+"/attr_triples_2") as f:
    for l in f:
        try:
            (e1, p, o) = l.rstrip("\n").split("\t")
        except ValueError:
            continue
        if e1 in ents_rel_2:
            new_attr_2.append((e1, p, o))

new_attr_1 = map(lambda x: x[0] + "\t" + x[1] + "\t" + x[2] + "\n", new_attr_1)
new_attr_2 = map(lambda x: x[0] + "\t" + x[1] + "\t" + x[2] + "\n", new_attr_2)

with open(path+"/attr_triples_1_new", "w") as f:
    f.writelines(new_attr_1)
with open(path+"/attr_triples_2_new", "w") as f:
    f.writelines(new_attr_2)
