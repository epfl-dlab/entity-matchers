import pickle
import sys
from collections import defaultdict, Counter

path = sys.argv[1]
save_file = sys.argv[2]
edges = []
nodes = set()
with open(path) as f:
    for l in f:
        (e1, _, e2) = l.rstrip("\n").split("\t")
        edges.append((e1, e2))
        nodes.add(e1)
        nodes.add(e2)
dict_in = defaultdict(int)
dict_out = defaultdict(int)
for (e1, e2) in edges:
    dict_out[e1] += 1
    dict_in[e2] += 1
dict_deg = {}
for e in nodes:
    in_e = dict_in[e]
    out_e = dict_out[e]
    dict_deg[e] = in_e+out_e

degs = list(dict_deg.values())
counts = Counter(degs)
with open(save_file, "wb") as f:
    pickle.dump(dict(counts.items()), f)
