from others.utils import *
from data_processor.file_io import *


def generate_v2(triples):
    ents_sorted = count_ent_degree(triples, is_sorted=True)
    avg_degree_original = 2*len(triples) / len(ents_sorted)
    print('%d*2 / %d = %.4f' % (len(triples), len(ents_sorted), 2*len(triples) / len(ents_sorted)))
    delete_ratio = 0.05
    # triples = random.sample(triples, int(len(triples)*0.8))
    while True:
        ents_to_delete = set(ents_sorted[-int(len(ents_sorted)*delete_ratio):])
        triples = set([(h, r, t) for (h, r, t) in triples if h not in ents_to_delete and t not in ents_to_delete])
        ents_sorted = count_ent_degree(triples, is_sorted=True)
        avg_degree_current = 2*len(triples) / len(ents_sorted)
        print('%d*2 / %d = %.4f' % (len(triples), len(ents_sorted), 2*len(triples) / len(ents_sorted)))
        if avg_degree_current >= 2*avg_degree_original:
            break
    return triples


if __name__ == '__main__':
    path = '/media/sl/Data/workspace/VLDB2019/SampKG/processed_data/rel_triples/rel_triples_WD_en'
    tr = generate_v2(read_triples(path))
    write_triples(path+'_V2', tr)
