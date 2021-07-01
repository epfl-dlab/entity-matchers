import argparse
import os
from tqdm import tqdm

def main(root_folder, dataset, keep_more):
    file_in = open(root_folder + "/" + dataset, "r")
    list_literals = []
    list_facts = []
    for line in tqdm(file_in):
        # if not line.startswith("<http://www.wikidata.org/"):
        #     continue
        # if line.startswith("<http://www.wikidata.org/value/") or line.startswith("<http://www.wikidata.org/reference/"):
        #     continue
        if "http://www.wikidata.org" not in line or "http://www.wikidata.org/value/" in line \
                or "http://www.wikidata.org/reference/" in line:
            continue
        if "wikiba" in line or "Special:EntityData" in line or "statement" in line:
            continue
        if not keep_more and ("<http://www.w3.org/2000/01/rdf-schema#label>" in line
                              or "<http://schema.org/name>" in line
                              or "<http://www.w3.org/2004/02/skos/core#altLabel>" in line):
            continue
        line = line.rstrip(".\n").split(" ")[:-1]
        h = line[0].replace("<", "").replace(">", "")
        r = line[1].replace("<", "").replace(">", "")
        t = " ".join(line[2:])
        if t.startswith('"'):
            if t.endswith('"') or t.endswith('>') or "prop" in r:
                list_literals.append("\t".join((h,r,t)) + "\n")
            elif t.endswith("@en"):
                list_literals.append("\t".join((h,r,t.rstrip("@en"))) + "\n")
        elif "prop" in r and "wikidata" in t:
            t = t.replace("<", "").replace(">", "")
            list_facts.append("\t".join((h,r,t)) + "\n")

    file_in.close()

    if not os.path.exists("{}/output/".format(root_folder)):
        os.mkdir("{}/output/".format(root_folder))

    if keep_more:
        suffix = "_more"
    else:
        suffix = ""

    with open("{}/output/attr_triples".format(root_folder) + suffix, 'w') as f:
        f.writelines(list_literals)

    with open("{}/output/rel_triples".format(root_folder) + suffix, 'w') as f:
        f.writelines(list_facts)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Automatically run the whole pipeline and compute results.")

    parser.add_argument(
        "--root_folder",
        type=str,
        help='Name of the root directory (it will be used even as output root.)'
    )

    parser.add_argument(
        "--dataset",
        help="Name of the file containing the Wikidata dataset",
        required=True
    )

    parser.add_argument(
        "--keep_more",
        help="Retain more literals (such as <http://schema.org/name> or <http://www.w3.org/2000/01/rdf-schema#label>)",
        required=False,
        action="store_true"
    )

    args_main = parser.parse_args()
    root_folder = args_main.root_folder
    dataset = args_main.dataset
    keep_more = args_main.keep_more
    main(root_folder, dataset, keep_more)
