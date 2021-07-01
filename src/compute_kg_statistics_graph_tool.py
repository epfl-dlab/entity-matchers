import graph_tool.all as gt
import time
import argparse

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

g = gt.Graph()
edges = []

with open("{}".format(args_main.graph), 'r') as f:
    for l in f:
        edges.append((l.split("\t")[0], l.split("\t")[2].rstrip("\n")))

g.add_edge_list(edges, hashed=True)

number_of_vertices = len(list(g.vertices()))
number_of_edges = len(list(g.edges()))

print("### STATISTICS ###")
print("\tNumber of vertices: {}".format(number_of_vertices))
print("\tNumber of edges: {}".format(number_of_edges))

clustering_coeff, std_error = gt.global_clustering(g)
avg_degree, avg_degree_std = gt.vertex_average(g, "total")
avg_in_degree, avg_in_degree_std = gt.vertex_average(g, "in")
avg_out_degree, avg_out_degree_std = gt.vertex_average(g, "out")

zero_in_deg = 0
zero_out_deg = 0
isolated_entities = 0
for v in g.vertices():
    if v.in_degree() == 0:
        zero_in_deg += 1
    if v.out_degree() == 0:
        zero_out_deg += 1
    if v.in_degree() == 0 and v.out_degree() == 0:
        isolated_entities += 1

print("\n\tClustering Coefficient: {}".format(clustering_coeff))
print("\tClustering Coefficient std: {}".format(std_error))

print("\n\tAverage degree: {}".format(avg_degree))
print("\tAverage degree std: {}".format(avg_degree_std))

print("\n\tAverage in degree: {}".format(avg_in_degree))
print("\tAverage in degree std: {}".format(avg_in_degree_std))

print("\n\tAverage out degree: {}".format(avg_out_degree))
print("\tAverage out degree std: {}".format(avg_out_degree_std))

print("\n\tIsolated entities: {}".format(isolated_entities))
print("\tPercentage isolated entities: {}".format(
    isolated_entities/number_of_vertices))
print("\n\tVertices with out degree zero: {}".format(zero_out_deg))
print("\tPercentage vertices with out degree zero: {}".format(
    zero_out_deg/number_of_vertices))

print("\n\tVertices with in degree zero: {}".format(zero_in_deg))
print("\tPercentage of vertices with in degree zero: {}".format(
    zero_in_deg/number_of_vertices))
print("\n____________________________________________________")
print("Total time: {} ".format(str(time.time() - start_time)))
