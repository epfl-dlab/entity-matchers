"""
Discriminator:

"""
from others.utils import *
import generator.strategy
from collections import defaultdict
from tqdm import tqdm


class Discriminator:
    def __init__(self, args, ddos):
        self.args = args
        self.strategy = eval('generator.strategy.' + args.target_dataset)
        self.max_degree1, self.max_degree2 = self.strategy['max_degree_kg1'], self.strategy['max_degree_kg2']
        self.ddo1, self.ddo2 = ddos[0], ddos[1]

    def accept_or_reject(self, sample_data, ent_links_full):
        sample_triples_1, sample_triples_2, sample_ent_links = sample_data[0], sample_data[1], sample_data[2]

        ents1 = set([e for (e, _, _) in sample_triples_1]) | set([e for (_, _, e) in sample_triples_1])
        ents2 = set([e for (e, _, _) in sample_triples_2]) | set([e for (_, _, e) in sample_triples_2])
        ent_num = self.args.ent_link_num

        sim_kg1 = js_divergence(self.ddo1, sample_triples_1, self.max_degree1)
        sim_kg2 = js_divergence(self.ddo2, sample_triples_2, self.max_degree2)
        print('sim_kg1:', sim_kg1)
        print('sim_kg2:', sim_kg2)
        print('len(ents1):', len(ents1))
        print('len(ents2):', len(ents2))

        truth1 = set([e for (e, _) in sample_ent_links])
        truth2 = set([e for (_, e) in sample_ent_links])
        in_truth1 = set([e for e in ents1 if e in truth1])
        in_truth2 = set([e for e in ents2 if e in truth2])
        out_truth1 = ents1.difference(in_truth1)
        out_truth2 = ents2.difference(in_truth2)
        print("Truth len:", len(sample_ent_links))
        print("Entities inside the truth KG1:", len(in_truth1))
        print("Entities inside the truth KG2:", len(in_truth2))
        print("Entities outside the truth KG1:", len(out_truth1))
        print("Entities outside the truth KG2:", len(out_truth2))

        # This is always 1:1 assumption: changed to only len of links
        if len(sample_ent_links) != ent_num or len(in_truth1) != ent_num or len(in_truth2) != ent_num:
            print('reject: num wrong')
            return False

        if sim_kg1 > self.args.js_expectation or sim_kg2 > self.args.js_expectation:
            print('reject: JS divergence not in ' + str(self.args.js_expectation))
            return False

        if not self.check_sample(sample_triples_1, sample_triples_2, ent_links_full, sample_ent_links):
            print('reject: entities outside the truth are matched')
            return False

        print('accept!')
        return True

    def check_sample(self, KG1_rel_triples, KG2_rel_triples, full_truth, sample_truth):
        kg1_truth = defaultdict(str)
        kg2_truth = defaultdict(str)
        out_truth = full_truth.difference(sample_truth)
        ents_kg1 = set([e for (e, _, _) in KG1_rel_triples]) | set([e for (_, _, e) in KG1_rel_triples])
        ents_kg2 = set([e for (e, _, _) in KG2_rel_triples]) | set([e for (_, _, e) in KG2_rel_triples])
        for (e1, e2) in tqdm(out_truth):
            if e1 in ents_kg1:
                kg1_truth[e1] = e2
            if e2 in ents_kg2:
                kg2_truth[e2] = e1
        for e1 in tqdm(ents_kg1):
            if kg1_truth[e1] in ents_kg2:
                print(e1, kg1_truth[e1])
                print("ERROR, entities outside the truth are matched from KG1 -> KG2")
                return False
        for e2 in tqdm(ents_kg2):
            if kg2_truth[e2] in ents_kg1:
                print(e2, kg2_truth[e2])
                print("ERROR, entities outside the truth are matched from KG2 -> KG1")
                return False
        print("OK, no 1:1 assumption")
        return True