import time
import argparse
from igraph import *

parser = argparse.ArgumentParser(
    description="Compute statistics of the graph specified as input. "
                "In particular, clustering coefficient, avg degree, avg "
                "in degree, avg out degree, isolated entities, "
                "entities with in/out degree == 0."
)

parser.add_argument(
    "--graph",
    type=str,
    help='File where the relations of the graph are stored'
)

args_main = parser.parse_args()

start_time = time.time()

edges = []

with open("{}".format(args_main.graph), 'r') as f:
    for l in f:
        edges.append((l.split("\t")[0], l.split("\t")[2].rstrip("\n")))
count = 0
dict_id = {}
edges_new = []
for (e1, e2) in edges:
    if e1 not in dict_id:
        dict_id[e1] = count
        count += 1
    if e2 not in dict_id:
        dict_id[e2] = count
        count += 1
    edges_new.append((dict_id[e1], dict_id[e2]))
g = Graph(directed=True, edges=edges_new)

number_of_vertices = g.vcount()
number_of_edges = g.ecount()

print("### STATISTICS ###")
print("\tNumber of vertices: {}".format(number_of_vertices))
print("\tNumber of edges: {}".format(number_of_edges))

clustering_coeff = g.transitivity_undirected()
degree = g.degree()
in_degree = g.degree(mode="in")
out_degree = g.degree(mode="out")

avg_degree = mean(degree)
avg_in_degree = mean(in_degree)
avg_out_degree = mean(out_degree)

print("\n\tClustering Coefficient: {}".format(clustering_coeff))

print("\n\tAverage degree: {}".format(avg_degree))

print("\n\tAverage in degree: {}".format(avg_in_degree))

print("\n\tAverage out degree: {}".format(avg_out_degree))

print("Total time: {} ".format(str(time.time() - start_time)))
