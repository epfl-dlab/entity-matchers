import random

from generator.entity_pagerank import PageRank
from others.utils import *
import generator.strategy


class Generator:

    def __init__(self, args, data):
        self.args = args
        self.triples_1, self.triples_2, self.ent_links = data.KG1_rel_triples, data.KG2_rel_triples, data.ent_links
        self.ents_deleted1, self.ents_deleted2 = data.ents_to_delete1, data.ents_to_delete2
        self.ents_to_keep1, self.ents_to_keep2 = data.ents_to_keep1, data.ents_to_keep2
        self.strategy = eval('generator.strategy.' + args.target_dataset)
        self.data = data
        self.generate_epoch = 1
        # self._generate_random()
        self._generate()

    def _generate_random(self):
        while len(self.ent_links) - self.args.ent_link_num > 150:
            delete_ratio = self.args.delete_random_ratio
            sample_num = int(len(self.ent_links) * (1 - delete_ratio) + self.args.ent_link_num * delete_ratio)
            print('delete num:', len(self.ent_links) - sample_num)
            self.ent_links = set(random.sample(self.ent_links, sample_num))
            self.check_links()
            self.print_log(func_name='random')
        while len(self.ent_links) > self.args.ent_link_num:
            self.ent_links = set(random.sample(self.ent_links, len(self.ent_links) - 1))
            self.check_links()
            self.print_log(func_name='random')
        self.sample_data = [self.triples_1, self.triples_2, self.ent_links]

    def _generate(self):
        args = self.args
        self.delete_limit = self.strategy['delete_limit']
        if args.init_speed != 0:
            # Delete by degree initially if we give it some speed
            self.delete_by_degree(is_print_log=True, delete_degree_ratio=args.init_speed)
        self.print_log(func_name='init')
        # self.ent_links = set(random.sample(self.ent_links, int(len(self.ent_links)*(1-self.strategy['delete_random_ratio']))))
        # self.check_links()
        # self.print_log(func_name='random')
        while len(self.ent_links) / args.ent_link_num > 1:
            rate = len(self.ent_links) / args.ent_link_num
            is_print_log = True
            if '15K' in args.target_dataset and rate > 1.5 or rate > 1.05:
                self.print_log()
                is_print_log = True
            if rate > self.delete_limit:
                self.delete_by_degree_distribution()
                self.delete_by_degree(is_print_log=True)
            else:
                self.delete_by_degree(is_print_log=is_print_log)

            self.check_links()
        self.print_log(func_name='delete_by_degree')
        if self.strategy['preserve_num'] != 0:
            self.preserve_high_degree_entity()
            self.preserve_high_degree_entity(is_change_position=True)
        while len(self.ent_links) / args.ent_link_num > 1:
            self.delete_by_degree(is_print_log=False)
        if 'YG_en_15K_V2' in args.target_dataset:
            self.triples_2 = delete_relation_yg(self.triples_2, self.data.ddo2, is_15K=True)
        if 'YG_en_100K_V2' in args.target_dataset:
            self.triples_2 = delete_relation_yg(self.triples_2, self.data.ddo2)
        self.print_log(func_name='delete_by_degree')

        self.sample_data = [self.triples_1, self.triples_2, self.ent_links]
        print(self.strategy)
        return

    def preserve_high_degree_entity(self, is_change_position=False):
        preserve_num = self.strategy['preserve_num']
        ents1 = set([e for (e, _, _) in self.triples_1]) | set([e for (_, _, e) in self.triples_1])
        ents2 = set([e for (e, _, _) in self.triples_2]) | set([e for (_, _, e) in self.triples_2])

        if is_change_position:
            ents_preserve2 = set()
            # Add all the ones with highest degree we can add
            for e in self.data.high_degree_ents2:
                if e not in ents2:
                    ents_preserve2.add(e)
                if len(ents_preserve2) >= preserve_num:
                    break
            links_preserve = set([(e1, e2) for (e1, e2) in self.data.ent_links if e2 in ents_preserve2])
            ents1_temp = set([e for (e, _) in links_preserve])
            ents_preserve1 = set()
            for e in ents1_temp:
                if e not in ents1:
                    ents_preserve1.add(e)
            ents_preserve2 = set([e2 for (e1, e2) in links_preserve if e1 in ents_preserve1])
        else:
            ents_preserve1 = set()
            for e in self.data.high_degree_ents1:
                if e not in ents1:
                    ents_preserve1.add(e)
                if len(ents_preserve1) >= preserve_num:
                    break
            links_preserve = set([(e1, e2) for (e1, e2) in self.data.ent_links if e1 in ents_preserve1])
            ents2_temp = set([e for (_, e) in links_preserve])
            ents_preserve2 = set()
            for e in ents2_temp:
                if e not in ents2:
                    ents_preserve2.add(e)
            ents_preserve1 = set([e1 for (e1, e2) in links_preserve if e2 in ents_preserve2])

        ents1 = ents1 | ents_preserve1
        ents2 = ents2 | ents_preserve2

        self.triples_1 = set([(h, r, t) for (h, r, t) in self.data.KG1_rel_triples if h in ents1 and t in ents1])
        self.triples_2 = set([(h, r, t) for (h, r, t) in self.data.KG2_rel_triples if h in ents2 and t in ents2])
        self.ent_links = set([(e1, e2) for (e1, e2) in self.data.ent_links if e1 in ents1 and e2 in ents2])
        self.print_log(func_name='preserve_high_degree_entity')
        return

    def delete_by_degree(self, min_priority=True, is_print_log=True, delete_degree_ratio=None):
        # delete entities with lowest degrees
        if delete_degree_ratio is None:
            delete_degree_ratio = self.strategy['delete_degree_ratio']
        if delete_degree_ratio == 0:
            return
        # size = how many entities left to remove
        size = len(self.ent_links) - self.args.ent_link_num  # ent_link_num is the final size
        delete_degree_num = max(int(size * delete_degree_ratio), 1)
        ents1_sorted = count_ent_degree(self.triples_1, is_sorted=True)
        ents2_sorted = count_ent_degree(self.triples_2, is_sorted=True)
        if min_priority:
            ents1_to_delete = set(ents1_sorted[-delete_degree_num:])
            ents2_to_delete = set()
            # when size reaches 5 delete_degree_num will be 1 so we delete 1 entites at a time from the left
            if size > 5:
                ents2_to_delete = set(ents2_sorted[-delete_degree_num:])
        else:
            ents1_sorted = ents1_sorted[-delete_degree_num * 10:]
            ents2_sorted = ents2_sorted[-delete_degree_num * 10:]
            random.shuffle(ents1_sorted)
            random.shuffle(ents2_sorted)
            ents1_to_delete = set(ents1_sorted[-delete_degree_num:])
            ents2_to_delete = set(ents2_sorted[-delete_degree_num:])
        self.update_triples_and_links(ents1_to_delete, ents2_to_delete)
        if is_print_log:
            self.print_log('delete_by_degree')
        return

    def delete_by_degree_distribution(self):
        delete_dd_ratio = self.strategy['delete_ratio']
        if delete_dd_ratio <= 0:
            return
        print("Deleting by PR/DD KG1")
        ents1_to_delete = self.delete_by_pagerank_for_dd(self.triples_1, self.data.ddo1, delete_dd_ratio,
                                                         self.strategy['max_degree_kg1'])
        print("Deleting by PR/DD KG2")
        ents2_to_delete = self.delete_by_pagerank_for_dd(self.triples_2, self.data.ddo2, delete_dd_ratio,
                                                         self.strategy['max_degree_kg2'])
        print("Entities to delete KG1:", len(ents1_to_delete))
        print("Entities to delete KG2:", len(ents2_to_delete))
        ents1_lk = set([e for (e, _) in self.ent_links])
        ents2_lk = set([e for (_, e) in self.ent_links])
        print("Entities to delete inside truth KG1:", len([e for e in ents1_to_delete if e in ents1_lk]))
        print("Entities to delete inside truth KG2:", len([e for e in ents2_to_delete if e in ents2_lk]))
        print("Entities to delete outside truth KG1:", len([e for e in ents1_to_delete if e in self.ents_to_keep1]))
        print("Entities to delete outside truth KG2:", len([e for e in ents2_to_delete if e in self.ents_to_keep2]))
        self.update_triples_and_links(ents1_to_delete, ents2_to_delete)
        self.print_log(func_name='delete_by_degree_distribution')
        return

    def delete_by_pagerank_for_dd(self, triples, ddo, delete_dd_ratio, max_degree):
        ents_to_delete = set()
        # Using nx instead of their pagerank
        print("Computing PR")
        page_rank = sorted(compute_pagerank(triples).items(), key=lambda p: p[1], reverse=False)
        ents_pr = [e for (e, _) in page_rank]
        print("Computing DD")
        ddc, degree_ents_dict = count_degree_distribution(triples, max_degree)
        delete_random_ratio = self.strategy['delete_random_ratio']
        print(format_print_dd(ddo, prefix='\t'))
        print(format_print_dd(ddc, prefix='\t'))
        for d, ents in degree_ents_dict.items():
            size = len(ents)
            if size == 0:
                continue
            # NOTE: delete_dd_ration is the mu parameter from the paper (see base step size) in the dsize
            delete_dd_num = int(size * delete_dd_ratio * (1 + ddc[d] - ddo[d]))
            if d < 8 and ddc[d] > ddo[d]:
                delete_dd_num = int(size * delete_dd_ratio * 3 * (1 + ddc[d] - ddo[d]))
            # delete_random_num: entity to delete random without considering pagerank
            delete_random_num = int(delete_dd_num * delete_random_ratio)
            # ents_to_delete_random: little "push" of entity to delete without considering rank
            ents_to_delete_random = set()
            if delete_random_num < size:
                ents_to_delete_random = set(random.sample(ents, delete_random_num))

            cnt = delete_random_num
            for e in ents_pr:
                if cnt >= delete_dd_num:
                    break
                if e in ents and e not in ents_to_delete_random:
                    ents_to_delete.add(e)
                    cnt += 1
            ents_to_delete.update(ents_to_delete_random)
        return ents_to_delete

    def update_triples_and_links(self, ents1_to_delete, ents2_to_delete):
        if self.args.open_dataset == 0:
            self.triples_1 = set([(h, r, t) for (h, r, t) in self.triples_1
                                  if h not in ents1_to_delete and t not in ents1_to_delete])
            self.triples_2 = set([(h, r, t) for (h, r, t) in self.triples_2
                                  if h not in ents2_to_delete and t not in ents2_to_delete])
        else:
            self.triples_1 = set([(h, r, t) for (h, r, t) in self.triples_1 if h not in ents1_to_delete])
            self.triples_2 = set([(h, r, t) for (h, r, t) in self.triples_2 if h not in ents2_to_delete])
        self.ent_links = set([(e1, e2) for (e1, e2) in self.ent_links if e1 not in ents1_to_delete
                              and e2 not in ents2_to_delete])
        self.ents_to_keep1 = set([e for e in self.ents_to_keep1 if e not in ents1_to_delete])
        self.ents_to_keep2 = set([e for e in self.ents_to_keep2 if e not in ents2_to_delete])
        self.check_links()
        return

    def check_links(self, debug = False):
        """
        It's no necessary to check entity links every epoch.
        :return:
        """
        ents1_lk = set([e for (e, _) in self.ent_links])
        ents2_lk = set([e for (_, e) in self.ent_links])
        ents1_tr = set([e for (e, _, _) in self.triples_1])
        ents2_tr = set([e for (e, _, _) in self.triples_2])
        if self.args.open_dataset == 0:
            ents1_tr = ents1_tr | set([e for (_, _, e) in self.triples_1])
            ents2_tr = ents2_tr | set([e for (_, _, e) in self.triples_2])
        ents1 = ents1_lk & ents1_tr
        ents2 = ents2_lk & ents2_tr
        ents1_lk_and_keep = ents1_lk.union(self.ents_to_keep1)
        ents2_lk_and_keep = ents2_lk.union(self.ents_to_keep2)
        if debug:
            print("Check links:")
            print("\tTotal entities to keep KG1:", len(self.ents_to_keep1))
            print("\tTotal entities to keep KG2:", len(self.ents_to_keep2))
            print("\tEntities KG1 & links:", len(ents1))
            print("\tEntities KG2 & links:", len(ents2))
            print("\tEntities KG1:", len(ents1_tr))
            print("\tEntities KG2:", len(ents2_tr))
            print("\tEntities KG1 links + keep:", len(ents1_lk_and_keep))
            print("\tEntities KG2 links + keep:", len(ents2_lk_and_keep))
            print("\tLinks num:", len(self.ent_links))
        stop = len(ents1_lk_and_keep - ents1_tr) \
               == len(ents1_tr - ents1_lk_and_keep) \
               == len(ents2_lk_and_keep - ents2_tr) \
               == len(ents2_tr - ents2_lk_and_keep) == 0
        if debug:
            print("\tTo have stop:")
            print("\tEntities KG1 links + keep - Entities KG1:", len(ents1_lk_and_keep - ents1_tr))
            print("\tEntities KG1 - Entities KG1 links + keep:", len(ents1_tr - ents1_lk_and_keep))
            print("\tEntities KG2 links + keep - Entities KG2:", len(ents2_lk_and_keep - ents2_tr))
            print("\tEntities KG2 - Entities KG2 links + keep:", len(ents2_tr - ents2_lk_and_keep))
            print("\tStop =", stop)
        if stop:
            if debug:
                print("OK!")
            return
        self.ent_links = set([(e1, e2) for (e1, e2) in self.ent_links if e1 in ents1 and e2 in ents2])
        if self.args.open_dataset == 0:
            ents1_and_keep = ents1.union(self.ents_to_keep1)
            ents2_and_keep = ents2.union(self.ents_to_keep2)
            self.triples_1 = set([(h, r, t) for (h, r, t) in self.triples_1
                                  if h in ents1_and_keep and t in ents1_and_keep])
            self.triples_2 = set([(h, r, t) for (h, r, t) in self.triples_2
                                  if h in ents2_and_keep and t in ents2_and_keep])
            self.ents_to_keep1 = self.ents_to_keep1 & ents1_tr
            self.ents_to_keep2 = self.ents_to_keep2 & ents2_tr
        else:
            self.triples_1 = set([(h, r, t) for (h, r, t) in self.triples_1 if h in ents1])
            self.triples_2 = set([(h, r, t) for (h, r, t) in self.triples_2 if h in ents2])
        self.check_links()

    def print_log(self, func_name=None):
        if func_name is None:
            print('\niteration:', self.generate_epoch)
            self.generate_epoch += 1
        else:
            ents1 = set([e for (e, _, _) in self.triples_1]) | set([e for (_, _, e) in self.triples_1])
            ents2 = set([e for (e, _, _) in self.triples_2]) | set([e for (_, _, e) in self.triples_2])
            ents1_lk = set([e for (e, _) in self.ent_links])
            ents2_lk = set([e for (_, e) in self.ent_links])
            ents1_out = set([e for e in ents1 if e not in ents1_lk])
            ents2_out = set([e for e in ents2 if e not in ents2_lk])
            print('\t' + func_name, ':')
            print('\tentity_num_1:', len(ents1))
            print('\tentity_num_2:', len(ents2))
            print('\tentity_out_1:', len(ents1_out))
            print('\tentity_out_2:', len(ents2_out))
            print('\ttriple_num_1:', len(self.triples_1))
            print('\ttriple_num_2:', len(self.triples_2))
            print('\tlink_num:', len(self.ent_links), '\n')
        return
