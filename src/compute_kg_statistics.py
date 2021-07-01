from collections import defaultdict, Counter
from tqdm import tqdm
import argparse
import networkx as nx
import pickle
import matplotlib.pyplot as plt


def compute_statistics(triples, dataset_name):
    edges = defaultdict(set)
    for h, _, t in tqdm(triples):
        edges[h].add(t)
    KG = nx.DiGraph(edges)
    print("Isolated entities:", len(list(nx.isolates(KG))))
    num_nodes = len(KG)
    print("Number of entities:", num_nodes)
    in0 = 0
    for _, deg in KG.in_degree():
        if deg == 0:
            in0 += 1
    print("Entities with in degree = 0:", in0)
    print("Percentage of entities with in degree = 0:", in0 / num_nodes)
    out0 = 0
    for _, deg in KG.out_degree():
        if deg == 0:
            out0 += 1
    print("Entities with out degree = 0:", out0)
    print("Percentage of entities with out degree = 0:", out0 / num_nodes)
    print("Average clustering coefficient:", nx.average_clustering(KG))
    print("Average degree:", len(KG.edges()) / len(KG))
    pr = nx.pagerank(KG, alpha=0.85)
    sort_pr = sorted(pr.items(), key=lambda p: p[1], reverse=True)
    print("Highest 20 pageranks:", sort_pr[:20])
    sort_pr2 = sorted(pr.items(), key=lambda p: p[1], reverse=False)
    print("Lowest 20 pageranks:", sort_pr2[:20])
    print("Saving PageRank...")
    with open("pr_" + dataset_name + "_.pkl", 'wb') as f:
        pickle.dump(pr, f)

    print("Saving degree distribution...")
    degree_sequence = sorted([d for n, d in KG.degree()], reverse=True)  # degree sequence
    degree_count = Counter(degree_sequence)
    deg, cnt = zip(*degree_count.items())
    plt.figure()
    plt.bar(deg, cnt)
    plt.title("Degree Log-Log Histogram")
    plt.ylabel("Count")
    plt.xlabel("Degree")
    plt.xscale("log")
    plt.yscale("log")
    plt.savefig("deg_hist_" + dataset_name + ".png")
    plt.figure()
    plt.plot(deg, cnt)
    plt.title("Degree Log-Log Plot")
    plt.ylabel("Count")
    plt.xlabel("Degree")
    plt.xscale("log")
    plt.yscale("log")
    plt.savefig("deg_plot_" + dataset_name + ".png")
    plt.figure()
    plt.bar(deg, cnt)
    plt.plot(deg, cnt)
    plt.title("Degree Log-Log Histogram Interpolated")
    plt.ylabel("Count")
    plt.xlabel("Degree")
    plt.xscale("log")
    plt.yscale("log")
    plt.savefig("deg_hist_inter_" + dataset_name + ".png")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Check KG statistics')
    parser.add_argument('--rel_triples', type=str, help="Path to the relation triples to consider")
    parser.add_argument('--dataset', type=str, help="Dataset name")
    args = parser.parse_args()
    triples = set()
    with open(args.rel_triples) as f:
        for l in tqdm(f):
            (e1, r, e2) = l.rstrip("\n").rstrip().split("\t")
            triples.add((e1, r, e2))
    compute_statistics(triples, args.dataset)
