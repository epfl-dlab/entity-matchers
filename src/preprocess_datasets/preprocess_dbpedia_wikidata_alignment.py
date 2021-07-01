import argparse
import os
from tqdm import tqdm


parser = argparse.ArgumentParser(
    description="Preprocess Alignment for DBpedia and Wikidata. First we print dbpedia, and then wikidata. "
)

parser.add_argument(
    "--alignment",
    type=str,
    help='Alignment file name.'
)

parser.add_argument(
    "--root_folder",
    type=str,
    help='Name of the root directory (it will be used even as output root.)'
)

parser.add_argument(
    "--output_suffix",
    type=str,
    help='Name of the output file will be ent_links_{suffix}'
)


if __name__ == "__main__":
    args_main = parser.parse_args()

    alignment = args_main.alignment
    suffix = args_main.output_suffix

    root_folder = args_main.root_folder

    list_alignments = []

    with open("{}/{}".format(root_folder, alignment), 'r') as f:
        for l in tqdm(f):
            if "www.wikidata.org" in l:
                line = (l.rstrip(".\n").replace("<", "").replace(">", "")).split(" ")
                list_alignments.append(line[0] + "\t" + line[2].strip() + "\n")

    if not os.path.exists("{}/output/".format(root_folder)):
        os.mkdir("{}/output/".format(root_folder))

    with open("{}/output/ent_links_{}".format(root_folder, suffix), 'w') as f:
        f.writelines(list_alignments)




