import sys

path = sys.argv[1]
edges = []
nodes = set()
with open(path) as f:
    for l in f:
        (e1, _, e2) = l.rstrip("\n").split("\t")
        edges.append((e1, e2))
        nodes.add(e1)
        nodes.add(e2)
print("Average degree:", float(len(edges))/float(len(nodes)))
