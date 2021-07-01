import random

from data_processor.file_io import *
from others.utils import *
import generator.strategy
from generator.entity_pagerank import PageRank
import pandas as pd
import numpy as np


class DataModel:

    def __init__(self, args):
        self.args = args
        self.strategy = eval('generator.strategy.' + args.target_dataset)
        self._init()

    def _init(self, debug=False):
        args = self.args
        self.ent_links_full = read_links(args.ent_link_path)
        self.ent_links = self.ent_links_full
        ents_kg1 = set([e for (e, _) in self.ent_links])
        ents_kg2 = set([e for (_, e) in self.ent_links])

        KG1_rel_triples_raw = read_triples(args.KG1_rel_triple_path)
        KG2_rel_triples_raw = read_triples(args.KG2_rel_triple_path)

        self.ddo1, _ = count_degree_distribution(KG1_rel_triples_raw, self.strategy['max_degree_kg1'])
        self.ddo2, _ = count_degree_distribution(KG2_rel_triples_raw, self.strategy['max_degree_kg2'])
        open_dataset = False
        if args.open_dataset == 1:
            open_dataset = True

        KG1_rel_triples_raw, ents_kg1 = filter_rel_triples_by_ents(KG1_rel_triples_raw, ents_kg1,
                                                                   open_dataset=open_dataset)
        KG2_rel_triples_raw, ents_kg2 = filter_rel_triples_by_ents(KG2_rel_triples_raw, ents_kg2,
                                                                   open_dataset=open_dataset)
        print("First filter entities by truth ")
        while len(self.ent_links) != len(ents_kg1) or len(self.ent_links) != len(ents_kg2):
            self.ent_links = set([(e1, e2) for (e1, e2) in self.ent_links if e1 in ents_kg1 and e2 in ents_kg2])
            ents_kg1 = set([e for (e, _) in self.ent_links])
            ents_kg2 = set([e for (_, e) in self.ent_links])
            KG1_rel_triples_raw, ents_kg1 = filter_rel_triples_by_ents(KG1_rel_triples_raw, ents_kg1,
                                                                       open_dataset=open_dataset)
            KG2_rel_triples_raw, ents_kg2 = filter_rel_triples_by_ents(KG2_rel_triples_raw, ents_kg2,
                                                                       open_dataset=open_dataset)
            if debug:
                print("Entities KG1:", len(ents_kg1))
                print("Entities KG2:", len(ents_kg2))
                print("Truth len:", len(self.ent_links))
                print("Entities KG1 from triples:",
                      len(set([e for (e, _, _) in KG1_rel_triples_raw]) | set(
                          [e for (_, _, e) in KG1_rel_triples_raw])))
                print("Entities KG2 from triples:",
                      len(set([e for (e, _, _) in KG2_rel_triples_raw]) | set(
                          [e for (_, _, e) in KG2_rel_triples_raw])))
        self.KG1_rel_triples = KG1_rel_triples_raw
        self.KG2_rel_triples = KG2_rel_triples_raw
        if self.strategy['preserve_num'] != 0:
            _, degree_ents_dict1 = count_degree_distribution(KG1_rel_triples_raw, self.strategy['max_degree_kg1'])
            _, degree_ents_dict2 = count_degree_distribution(KG2_rel_triples_raw, self.strategy['max_degree_kg2'])
            # All the ones with highest degree (i.e. 100)
            self.high_degree_ents1 = set(degree_ents_dict1[self.strategy['max_degree_kg1']])
            self.high_degree_ents2 = set(degree_ents_dict2[self.strategy['max_degree_kg2']])

        # # TODO: why two times?
        # self.KG1_rel_triples, ents_kg1 = filter_rel_triples_by_ents(KG1_rel_triples_raw, ents_kg1, open_dataset=open_dataset)
        # self.KG2_rel_triples, ents_kg2 = filter_rel_triples_by_ents(KG2_rel_triples_raw, ents_kg2, open_dataset=open_dataset)

        if args.pre_delete == 1 or args.pre_delete == 3:
            ents_kg1 = self.pre_kg(self.KG1_rel_triples)
            self.KG1_rel_triples, ents_kg1 = filter_rel_triples_by_ents(self.KG1_rel_triples, ents_kg1,
                                                                        open_dataset=open_dataset)
        if args.pre_delete == 2 or args.pre_delete == 3:
            delete_ratio = 1.0
            if 'DBP_en_YG_en_15K_V1' in args.target_dataset:
                delete_ratio = 0.95
            ents_kg2 = self.pre_kg(self.KG2_rel_triples, delete_ratio=delete_ratio)
            self.KG2_rel_triples, ents_kg2 = filter_rel_triples_by_ents(self.KG2_rel_triples, ents_kg2,
                                                                        open_dataset=open_dataset)

        self.KG1_attr_triples = filter_attr_triples_by_ents(read_triples(args.KG1_attr_triple_path), ents_kg1)
        self.KG2_attr_triples = filter_attr_triples_by_ents(read_triples(args.KG2_attr_triple_path), ents_kg2)

        print("Entities before deleting:", len(ents_kg1), len(ents_kg2))
        ents_kg1, ents_kg2 = self.remove_by_pagerank(ents_kg1, ents_kg2, debug)
        self.KG1_attr_triples = filter_attr_triples_by_ents(self.KG1_attr_triples, ents_kg1)
        self.KG2_attr_triples = filter_attr_triples_by_ents(self.KG2_attr_triples, ents_kg2)

        print('rel_triples:', len(self.KG1_rel_triples), len(self.KG2_rel_triples))
        print('attr_triples:', len(self.KG1_attr_triples), len(self.KG2_attr_triples))
        print('Entities after deleting:', len(ents_kg1), len(ents_kg2))
        print('Entities to keep KG1 and KG2:', len(self.ents_to_keep1), len(self.ents_to_keep2))
        print('Links size:', len(self.ent_links))
        return

    def write_generated_data(self, sample_data, sample_index):
        rel_triples_1, rel_triples_2, ent_links = sample_data[0], sample_data[1], sample_data[2]

        output_folder = self.args.output_folder + str(sample_index) + '/'
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
        output_folder += self.args.target_dataset + '/'
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        if self.args.draw_degree_distribution:
            self.draw(sample_data, output_folder)

        ents_kg1 = set([e for (e, _, _) in rel_triples_1]) | set([e for (_, _, e) in rel_triples_1])
        ents_kg2 = set([e for (e, _, _) in rel_triples_2]) | set([e for (_, _, e) in rel_triples_2])

        write_links(output_folder + 'ent_links', ent_links)
        split_and_write_entity_links(ent_links, output_folder)

        write_triples(output_folder + 'rel_triples_1', rel_triples_1)
        write_triples(output_folder + 'rel_triples_2', rel_triples_2)
        write_triples(output_folder + 'attr_triples_1', filter_attr_triples_by_ents(self.KG1_attr_triples, ents_kg1))
        write_triples(output_folder + 'attr_triples_2', filter_attr_triples_by_ents(self.KG2_attr_triples, ents_kg2))
        return

    def remove_by_pagerank(self, ents_kg1, ents_kg2, debug=False):
        print("Computing PR for KG1...")
        pr1 = pd.DataFrame(compute_pagerank(self.KG1_rel_triples).items(), columns=["KG1", "PR1"])
        print("Computing PR for KG2...")
        pr2 = pd.DataFrame(compute_pagerank(self.KG2_rel_triples).items(), columns=["KG2", "PR2"])
        ents_df = pd.DataFrame(self.ent_links, columns=["KG1", "KG2"])
        final_df = ents_df.merge(pr1, left_on="KG1", right_on="KG1").merge(pr2, left_on="KG2", right_on="KG2")
        if debug:
            final_df.to_csv("final_df_rank_or_" + self.args.target_dataset + "_.csv")
        final_df["PR1"] = final_df[["PR1"]] * self.args.delete_param * len(ents_kg1)
        final_df["PR2"] = final_df[["PR2"]] * self.args.delete_param * len(ents_kg2)
        final_df["avg"] = final_df[['PR1', 'PR2']].mean(axis=1)
        if debug:
            final_df.to_csv("final_df_rank_mult_" + self.args.target_dataset + "_.csv")
        self.ents_to_delete1 = set()
        self.ents_to_delete2 = set()
        self.ents_to_keep1 = set()
        self.ents_to_keep2 = set()
        for i in np.arange(0.1, 1.1, 0.1):
            print("Total entities with AVG PR (scaled) > {}:".format(i), len(final_df[final_df.avg > i]))
        print("Deleting entities by PR")
        for _, row in tqdm(final_df.iterrows()):
            if np.random.random() >= row["avg"]:
                if np.random.random() <= 0.5:
                    self.ents_to_delete1.add(row["KG1"])
                    self.ents_to_keep2.add(row["KG2"])
                else:
                    self.ents_to_delete2.add(row["KG2"])
                    self.ents_to_keep1.add(row["KG1"])

        print("After deleting first time:")
        print("\tTotal entities to delete KG1:", len(self.ents_to_delete1))
        print("\tTotal entities to keep KG2:", len(self.ents_to_keep2))
        print("\tTotal entities to delete KG2:", len(self.ents_to_delete2))
        print("\tTotal entities to keep KG1:", len(self.ents_to_keep1))
        self.KG1_rel_triples = set([(h, r, t) for (h, r, t) in self.KG1_rel_triples
                                    if h not in self.ents_to_delete1 and t not in self.ents_to_delete1])
        self.KG2_rel_triples = set([(h, r, t) for (h, r, t) in self.KG2_rel_triples
                                    if h not in self.ents_to_delete2 and t not in self.ents_to_delete2])
        ents_kg1_new = set([e for (e, _, _) in self.KG1_rel_triples]) | set([e for (_, _, e) in self.KG1_rel_triples])
        ents_kg2_new = set([e for (e, _, _) in self.KG2_rel_triples]) | set([e for (_, _, e) in self.KG2_rel_triples])
        self.ents_to_keep1 = self.ents_to_keep1.intersection(ents_kg1_new)
        self.ents_to_keep2 = self.ents_to_keep2.intersection(ents_kg2_new)
        self.ent_links = set([(e1, e2) for (e1, e2) in self.ent_links if e1 in ents_kg1_new and e2 in ents_kg2_new])
        if debug:
            print("After first filter:")
            print("\tTotal entities to keep KG1:", len(self.ents_to_keep1))
            print("\tTotal entities to keep KG2:", len(self.ents_to_keep2))
            print("\tEntities KG1:", len(ents_kg1_new))
            print("\tEntities KG2:", len(ents_kg2_new))
            print("\tLinks num:", len(self.ent_links))
        while (len(self.ent_links) + len(self.ents_to_keep1) != len(ents_kg1_new) or
               len(self.ent_links) + len(self.ents_to_keep2) != len(ents_kg2_new)):
            ents1_lk = set([e for (e, _) in self.ent_links])
            ents2_lk = set([e for (_, e) in self.ent_links])
            ents1_lk_and_keep = ents1_lk.union(self.ents_to_keep1)
            ents2_lk_and_keep = ents2_lk.union(self.ents_to_keep2)
            self.KG1_rel_triples = set([(h, r, t) for (h, r, t) in self.KG1_rel_triples
                                        if h in ents1_lk_and_keep
                                        and t in ents1_lk_and_keep])
            self.KG2_rel_triples = set([(h, r, t) for (h, r, t) in self.KG2_rel_triples
                                        if h in ents2_lk_and_keep
                                        and t in ents2_lk_and_keep])
            ents_kg1_new = set([e for (e, _, _) in self.KG1_rel_triples]) | set(
                [e for (_, _, e) in self.KG1_rel_triples])
            ents_kg2_new = set([e for (e, _, _) in self.KG2_rel_triples]) | set(
                [e for (_, _, e) in self.KG2_rel_triples])
            self.ents_to_keep1 = self.ents_to_keep1.intersection(ents_kg1_new)
            self.ents_to_keep2 = self.ents_to_keep2.intersection(ents_kg2_new)
            self.ent_links = set([(e1, e2) for (e1, e2) in self.ent_links if e1 in ents_kg1_new and e2 in ents_kg2_new])
            if debug:
                print("Iterating:")
                print("\tTotal entities to keep KG1:", len(self.ents_to_keep1))
                print("\tTotal entities to keep KG2:", len(self.ents_to_keep2))
                print("\tEntities KG1:", len(ents_kg1_new))
                print("\tEntities KG2:", len(ents_kg2_new))
                print("\tLinks num:", len(self.ent_links))
        if debug:
            print("OK!")
        return ents_kg1_new, ents_kg2_new

    @staticmethod
    def pre_kg(triples, delete_ratio=1.0):
        _, degree_ents_dict = count_degree_distribution(triples, 100)
        degree_ents = list(degree_ents_dict.values())
        ents_to_sample = set()
        # Seen that order is random here
        for i in range(-10, 0):
            ents_to_sample = ents_to_sample | set(degree_ents[i])
        ents_to_delete = set(random.sample(ents_to_sample, int(len(ents_to_sample) * delete_ratio)))
        ents = set([h for (h, _, _) in triples]) | set([t for (_, _, t) in triples])
        return ents - ents_to_delete

    def draw(self, sample_data, output_folder):
        rel_triples_1, rel_triples_2 = sample_data[0], sample_data[1]

        dd_sample1, _ = count_degree_distribution(rel_triples_1, 100)
        dd_sample2, _ = count_degree_distribution(rel_triples_2, 100)

        cdf_kg1 = count_cdf(self.ddo1, 100)
        cdf_kg2 = count_cdf(self.ddo2, 100)
        cdf_sample1 = count_cdf(dd_sample1, 100)
        cdf_sample2 = count_cdf(dd_sample2, 100)
        print(format_print_dd(self.ddo1, prefix='dd_kg1:    \t'))
        print(format_print_dd(dd_sample1, prefix='dd_sample1:\t'))
        print(format_print_dd(self.ddo2, prefix='dd_kg2:    \t'))
        print(format_print_dd(dd_sample2, prefix='dd_sample2:\t'))
        draw_fig([cdf_kg1, cdf_sample1, cdf_kg2, cdf_sample2],
                 ['source KG 1', 'sampled data_processor 1', 'source KG 2', 'sampled data_processor 2'],
                 [output_folder + 'cdf_' + self.args.target_dataset, 'degree', 'cdf'])
        draw_fig([self.ddo1, dd_sample1, self.ddo2, dd_sample2],
                 ['source KG 1', 'sampled data_processor 1', 'source KG 2', 'sampled data_processor 2'],
                 [output_folder + 'degree_distribution_' + self.args.target_dataset, 'degree', 'degree distribution'],
                 limit=[0, 20, 0, 0.4])
        return
