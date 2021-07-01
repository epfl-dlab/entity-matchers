import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Remove extra attributes from a pair of datasets. '
                                                 'NOTE: This will replace the attribute file! Do it in another folder')
    parser.add_argument("--dataset", type=str, help="Path to dataset folder")
    parser.add_argument("--openea_dataset", type=str, help="Path to OpenEA dataset to take attributes from")
    args = parser.parse_args()

    attr1 = []
    p1 = set()
    with open(args.dataset + "attr_triples_1") as f:
        for l in f:
            (e, p, o) = l.rstrip("\n").split("\t")
            p1.add(p)
            attr1.append((e, p, o))
    p1_openea = set()
    with open(args.openea_dataset + "attr_triples_1") as f:
        for l in f:
            (e, p, o) = l.rstrip("\n").split("\t")
            p1_openea.add(p)
    attr2 = []
    p2 = set()
    with open(args.dataset + "attr_triples_2") as f:
        for l in f:
            (e, p, o) = l.rstrip("\n").split("\t")
            p2.add(p)
            attr2.append((e, p, o))
    p2_openea = set()
    with open(args.openea_dataset + "attr_triples_2") as f:
        for l in f:
            (e, p, o) = l.rstrip("\n").split("\t")
            p2_openea.add(p)

    if "DBP" in args.dataset:
        p1_openea.remove("http://purl.org/dc/elements/1.1/description")
        p1_openea.add("http://dbpedia.org/ontology/description")
    p2_openea_new = set()
    if "WD" in args.dataset:
        for p in p2_openea:
            if "entity" in p:
                p2_openea_new.add(p.replace("entity", "prop/direct"))
            else:
                p2_openea_new.add(p)
        p2_openea = p2_openea_new

    final_p1 = p1 & p1_openea
    final_p2 = p2 & p2_openea
    with open(args.dataset + "attr_triples_1", "w") as f:
        for (e, p, o) in attr1:
            if p in final_p1:
                f.write(e + "\t" + p + "\t" + o + "\n")
    with open(args.dataset + "attr_triples_2", "w") as f:
        for (e, p, o) in attr2:
            if p in final_p2:
                f.write(e + "\t" + p + "\t" + o + "\n")
