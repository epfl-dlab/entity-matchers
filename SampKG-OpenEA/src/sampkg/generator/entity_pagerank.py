import time


class PageRank:
    def __init__(self, triples):
        self.damping_factor = 0.85
        self.max_iterations = 100
        self.min_delta = 5e-21

        self.triples = triples
        self.link_adjacency_list = self._count_link_adjacency_with_weight()
        self.entity_page_rank = self._compute_entity_page_rank()
        page_rank = sorted(self.entity_page_rank.items(), key=lambda d: d[1], reverse=False)
        print("Lowest pageranks:", page_rank[:50])
        # ordered from the smallest to largest
        self.page_rank = [e for (e, _) in page_rank]

    def _compute_entity_page_rank(self):
        start_time = time.time()
        ents = set([e for (e, _, _) in self.triples]) | set([e for (_, _, e) in self.triples])
        size = len(ents)
        page_rank = {}
        for e in ents:
            page_rank[e] = 1.0/size

        damping_value = (1.0 - self.damping_factor) / size

        flag = False
        iter_cnt = 0
        for iter_cnt in range(self.max_iterations):
            change = 0
            for h, adjacency_list in self.link_adjacency_list.items():
                rank = 0
                neighbor_size = len(adjacency_list)
                for t in adjacency_list:
                    rank += self.damping_factor * (page_rank[t] / neighbor_size)
                rank += damping_value
                change += abs(page_rank[h] - rank)
                page_rank[h] = rank

            if change < self.min_delta:
                flag = True
                break

        if flag:
            print("\tfinished in %s iterations!" % (iter_cnt+1))
        else:
            print("\tfinished out of 100 iterations!", change)
        print('\trun time: %.2f s' % (time.time()-start_time))
        return page_rank

    def _count_link_adjacency_with_weight(self):
        link_adjacency = {}
        for (h, _, t) in self.triples:
            adj_list = set()
            if t in link_adjacency:
                adj_list = link_adjacency[t]
            adj_list.add(h)
            link_adjacency[t] = adj_list
        return link_adjacency

















