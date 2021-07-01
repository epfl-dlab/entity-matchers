import sys
path = sys.argv[1]
if 'DBP_en_YG' in path:
    name_attribute_list = {'http://dbpedia.org/ontology/birthName', 'skos:prefLabel'}
elif 'DBP_en_WD' in path:
    name_attribute_list = {'http://dbpedia.org/ontology/birthName', 'http://www.w3.org/2004/02/skos/core#prefLabel'}
new_attr_1 = {}
new_attr_2 = {}
with open(path + "/attr_triples_1") as f:
    for l in f:
        (e, p, o) = l.rstrip("\n").split("\t")
        if p in name_attribute_list:
            new_attr_1[e] = (p, o)
        elif e not in new_attr_1:
            new_attr_1[e] = ("RDGCN:label", e.split('/')[-1].replace('_', ' '))
with open(path + "/attr_triples_2") as f:
    for l in f:
        (e, p, o) = l.rstrip("\n").split("\t")
        if p in name_attribute_list:
            new_attr_2[e] = (p, o)
        elif e not in new_attr_2:
            new_attr_2[e] = ("RDGCN:label", e.split('/')[-1].replace('_', ' '))

attr_1 = [x[0] + "\t" + x[1][0] + "\t" + x[1][1] + "\n" for x in new_attr_1.items()]
attr_2 = [x[0] + "\t" + x[1][0] + "\t" + x[1][1] + "\n" for x in new_attr_2.items()]
with open(path + "/attr_triples_1_new", "w") as f:
    f.writelines(attr_1)
with open(path + "/attr_triples_2_new", "w") as f:
    f.writelines(attr_2)
