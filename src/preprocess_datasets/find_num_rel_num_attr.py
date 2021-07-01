import sys

path = sys.argv[1]
ents_1 = set()
ents_2 = set()
rels_1 = set()
rels_2 = set()
attr_1 = set()
attr_2 = set()
prop_1 = set()
prop_2 = set()
links_1 = set()
links_2 = set()
print("Reading rel_triples_1")
with open(path + "/rel_triples_1") as f:
    for l in f:
        (e1, r, e2) = l.rstrip("\n").split("\t")
        rels_1.add(r)
        ents_1.add(e1)
        ents_1.add(e2)
print("Reading rel_triples_2")
with open(path + "/rel_triples_2") as f:
    for l in f:
        (e1, r, e2) = l.rstrip("\n").split("\t")
        rels_2.add(r)
        ents_2.add(e1)
        ents_2.add(e2)
print("Reading attr_triples_1")
with open(path + "/attr_triples_1") as f:
    for l in f:
        (_, p, o) = l.rstrip("\n").split("\t")
        prop_1.add(p)
        attr_1.add(o)
print("Reading attr_triples_2")
with open(path + "/attr_triples_2") as f:
    for l in f:
        (_, p, o) = l.rstrip("\n").split("\t")
        prop_2.add(p)
        attr_2.add(o)
print("Reading ent_links")
with open(path + "/ent_links") as f:
    for l in f:
        (e1, e2) = l.rstrip("\n").split("\t")
        links_1.add(e1)
        links_2.add(e2)
num_in_1 = 0
num_in_2 = 0
num_out_1 = 0
num_out_2 = 0
print("Counting entities in and out")
for e in ents_1:
    if e in links_1:
        num_in_1 += 1
    else:
        num_out_1 += 1
for e in ents_2:
    if e in links_2:
        num_in_2 += 1
    else:
        num_out_2 += 1
print("Entities 1:", len(ents_1))
print("Entities 2:", len(ents_2))
print("Num relations 1:", len(rels_1))
print("Num relations 2:", len(rels_2))
print("Num properties 1:", len(prop_1))
print("Num properties 2:", len(prop_2))
print("Num attributes 1:", len(attr_1))
print("Num attributes 2:", len(attr_2))
print("Num in 1:", num_in_1)
print("Num in 2:", num_in_2)
print("Num out 1:", num_out_1)
print("Num out 2:", num_out_2)
print("Prop. out 1:", float(num_out_1)/float(len(ents_1)))
print("Prop. out 2:", float(num_out_2)/float(len(ents_2)))

