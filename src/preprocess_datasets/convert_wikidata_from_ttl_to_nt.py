import rdflib
from rdflib import Graph
from tqdm import tqdm
import argparse


parser = argparse.ArgumentParser(
    description="Convert wikidata from ttl format to nt format using rdflib."
)

parser.add_argument(
    "--ttl_input_file",
    type=str,
    help='Path to wikidata .ttl already uncompressed file.'
)

parser.add_argument(
    "--nt_output_file",
    type=str,
    help='Output file for the .nt file'
)


if __name__ == "__main__":
    args_main = parser.parse_args()
    ttl_input_file = args_main.ttl_input_file
    nt_output_file = args_main.nt_output_file

    # Create the graph
    g = Graph()
    g.parse(ttl_input_file, format="turtle")

    print('Graph parsed, it contains {} alignments'.format(len(g)))

    with open(nt_output_file, 'w') as f:
        f.write(g.serialize(format='nt').decode("utf-8"))

    print('Graph serialized successfully in {}'.format(nt_output_file))
