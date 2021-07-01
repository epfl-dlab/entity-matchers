import matplotlib.pyplot as plt
import scipy.stats
import numpy as np
import networkx as nx
from collections import defaultdict


def draw_fig(num_list, line_names, labels, limit=None):
    """

    :param num_list: e.g. [[1, 2, 3], [4, 5, 6]]
    :param line_names: e.g. ['line1', 'line2']
    :param labels: e.g. ['fig_name', 'x', 'y']
    :param limit: e.g. [0, 10, 5, 20] means x:0->10 and y:2->20
    :return:
    """
    if limit is None:
        limit = [0, 40, 0, 1.0]

    color = ['b', 'b--', 'r', 'r--']
    for i in range(len(num_list)):
        plt.plot(range(len(num_list[i])), num_list[i], color[i], label=line_names[i])

    plt.xlim(limit[0], limit[1])
    plt.ylim(limit[2], limit[3])
    plt.ylabel(labels[2])
    plt.xlabel(labels[1])
    plt.title(labels[0].split('/')[-1])
    plt.legend()
    plt.savefig(labels[0] + '.png')
    plt.close()
    # plt.show()
    return


def count_degree_distribution(triples, max_degree):
    ent_degree = count_ent_degree(triples)
    dd = [0 for _ in range(0, max_degree + 1)]
    degree_ents_dict = {}
    for e, d in ent_degree.items():
        d = min(max_degree, d)
        # TODO: why truncating degree?
        dd[d] += 1
        ents = set()
        if d in degree_ents_dict:
            ents = degree_ents_dict[d]
        ents.add(e)
        degree_ents_dict[d] = ents
    dd_sum = sum(dd)
    dd = [d / dd_sum for d in dd]
    return dd, degree_ents_dict


def count_cdf(dd, max_degree):
    cdf = [0 for _ in range(max_degree + 1)]
    sum_temp = 0
    for i in range(len(dd)):
        sum_temp += dd[i]
        cdf[i] = sum_temp
    return cdf


def count_ent_degree(triples, is_sorted=False):
    ent_degree = {}
    for (h, _, t) in triples:
        degree = 1
        if h in ent_degree:
            degree += ent_degree[h]
        ent_degree[h] = degree

        degree = 1
        if t in ent_degree:
            degree += ent_degree[t]
        ent_degree[t] = degree
    if is_sorted:
        ent_degree = sorted(ent_degree.items(), key=lambda d: d[1], reverse=True)
        # return list of entity sorted by highest degree
        return [e for (e, _) in ent_degree]
    return ent_degree


def filter_attr_triples_by_ents(triples, ents):
    return set([(s, p, o) for (s, p, o) in triples if s in ents])


def filter_rel_triples_by_ents(triples, ents, open_dataset=False):
    # This is where the triples are filtered and only the entities in the ground truth remain
    rel_triples_new = set()
    ents_new = set()
    for (h, r, t) in triples:
        if h in ents and (t in ents or open_dataset):
            rel_triples_new.add((h, r, t))
            ents_new.add(h)
            ents_new.add(t)
    return rel_triples_new, ents_new


def format_print_dd(dd, prefix=''):
    output = prefix
    for i in dd:
        i = str(round(i, 6))
        if '.' in i:
            while len(i[i.index('.') + 1:]) < 6:
                i += '0'
        output += i + '  '
    return output


def js_divergence(ddo, sample_triples, max_degree=100):
    ddc, _ = count_degree_distribution(sample_triples, max_degree)
    ddo = np.array(ddo)
    ddc = np.array(ddc)
    m = (ddo + ddc) / 2
    return 0.5 * scipy.stats.entropy(ddo, m) + 0.5 * scipy.stats.entropy(ddc, m)


def delete_relation_yg(triples, ddo, is_15K=False):
    cnt = 10
    while True:
        if is_15K:
            indexs = [5, 6]
        else:
            indexs = [cnt, cnt + 15]
            cnt -= 1
        triples_new, ents = set(), set()
        ddc, degree_ents = count_degree_distribution(triples, 100)
        for d in range(indexs[0], indexs[1]):
            if d in degree_ents:
                ents = ents | set(degree_ents[d])
        e_delete = set()
        for (h, r, t) in triples:
            if h in ents and t in ents and h not in e_delete and t not in e_delete:
                e_delete.add(h)
                e_delete.add(t)
                continue
            triples_new.add((h, r, t))
        ddc, _ = count_degree_distribution(triples, 100)
        if js_divergence(ddo, triples_new) < 0.05 or cnt <= 5:
            break
        triples = triples_new
    return triples_new


def compute_pagerank(triples):
    edges = defaultdict(set)
    for (h, _, t) in triples:
        edges[h].add(t)
    KG = nx.DiGraph(edges)
    pr = nx.pagerank(KG, alpha=0.85)
    return pr
