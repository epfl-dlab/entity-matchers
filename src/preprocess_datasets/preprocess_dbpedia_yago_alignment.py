import argparse
import os
from tqdm import tqdm


parser = argparse.ArgumentParser(
    description="Preprocess Alignment for DBpedia and Yago. First we print dbpedia, and then yago. "
                "In case you specify align_entities argument, it will print only the alignments for entities "
                "that are present in both datasets."
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

parser.add_argument(
    "--align_entities",
    type=bool,
    help='Set to True if you want to output only the alignments that are present in both datasets.'
)

parser.add_argument(
    "--dbpedia_folder",
    type=str,
    help='Name of the directory where attr_triples and rel_triples are kept for dbpedia. Makes sense only when '
         'align_entities is True'
)

parser.add_argument(
    "--yago_folder",
    type=str,
    help='Name of the directory where attr_triples and rel_triples are kept for dbpedia. Makes sense only when '
         'align_entities is True'
)


def set_of_entities(folder: str) -> set:
    """
    Run through both files (rel_triples and attr_triples) contained in the folder,
    and return the set of all entities
    Parameters
    ----------
    folder

    Returns
    -------
    entities: a set with all entities contained both in attr_triples and rel_triples
    """
    entities = set()
    with open(folder + "/attr_triples", 'r') as f:
        for l in f:
            # Add all entities in attr_triples (only the first element is an entity)
            entities.add(l.split("\t")[0])

    with open(folder + "/rel_triples", 'r') as f:
        for l in f:
            # Add all entities in attr_triples (the first and third element are entities)
            entities.add(l.split("\t")[0].strip())
            entities.add(l.split("\t")[2].rstrip('\n').strip())

    print("There are {} unique entities in {}".format(
        len(entities), folder
    ))
    return entities


if __name__ == "__main__":
    args_main = parser.parse_args()

    alignment = args_main.alignment
    suffix = args_main.output_suffix

    root_folder = args_main.root_folder

    align_entities = args_main.align_entities
    dbpedia_folder = args_main.dbpedia_folder
    yago_folder = args_main.yago_folder

    list_alignments = []

    if align_entities and (not yago_folder or not dbpedia_folder):
        print("If you want to align entities based on the entities you have in your dataset, "
              "you need to specify yago_folder and dbpedia_folder arguments")
        exit(0)

    ent_yago = set()
    ent_dbpedia = set()

    if align_entities:
        # Create the set of entities
        ent_yago = set_of_entities(yago_folder)
        ent_dbpedia = set_of_entities(dbpedia_folder)

    with open("{}/{}".format(root_folder, alignment), 'r') as f:
        for l in tqdm(f):
            line = (l.rstrip(".\n").replace("<", "").replace(">", "")).split("\t")
            if not align_entities or (line[2].strip() in ent_dbpedia and line[0] in ent_yago):
                # Add to result only if we do not care about the alignment,
                # or when we have both entities in our dataset.
                list_alignments.append(line[2].strip() + "\t" + line[0] + "\n")

    if not os.path.exists("{}/output/".format(root_folder)):
        os.mkdir("{}/output/".format(root_folder))

    with open("{}/output/ent_links_{}".format(root_folder, suffix), 'w') as f:
        f.writelines(list_alignments)




