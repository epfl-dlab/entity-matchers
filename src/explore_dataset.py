import argparse
from tqdm import tqdm


parser = argparse.ArgumentParser(
    description="Check how many entities are missing from the large wikidata dataset."
)

parser.add_argument(
    "--large_dataset",
    help='directory of the large .nt file to be checked.'
)
parser.add_argument(
    "--small_dataset",
    help='directory of the small .nt file to be checked.'
)

if __name__ == '__main__':
    args_main = parser.parse_args()

    ds_large = args_main.large_dataset
    ds_small = args_main.small_dataset

    entity_set = {}
    # First load small dataset into a set of booleans.
    with open(ds_small, 'r') as f:
        for l in f:
            # Add to the set
            entity_set[l.split("entity/Q")[1].split()[0]] = False

    with open(ds_large, 'r') as f:
        # print("number of lines in f {}".format(len(f)))
        for l in tqdm(f):
            # mark as present
            try:
                if l.split("entity/Q")[1].split(">")[0] in entity_set:
                    entity_set[l.split("entity/Q")[1].split(">")[0]] = True
            except IndexError:
                pass

    counter_not_present = 0
    for k in entity_set:
        if entity_set[k] is False:
            counter_not_present += 1
    print("number of entities not present: {}".format(counter_not_present))

