from file_io import *


def read_triple_dbp_raw(file_path, language):
    triples = set()
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip(' .\n').split('> ')
            if len(line) != 3:
                continue
            # print(line)
            s = line[0].lstrip('<')
            p = line[1].lstrip('<')
            o = line[2].lstrip('<').rstrip('>')
            if '"@'+language in o:
                o = o.lstrip('"').rstrip('"@'+language)
            # print(s, p, o)
            triples.add((s, p, o))
    print('read raw triple dbp:', len(triples))
    return triples


def run_dbp(language):
    folder = 'H:\workspace\VLDB2020\SampKG\\'
    rel_triples = read_triple_dbp_raw(folder+'raw_data\DBpedia\\mappingbased_objects_'+language+'.ttl', language)
    attr_triples = read_triple_dbp_raw(folder+'raw_data\DBpedia\\mappingbased_literals_'+language+'.ttl', language)
    write_triples(folder+'processed_data\\rel_triples\\rel_triples_DBP_'+language, rel_triples)
    write_triples(folder+'processed_data\\attr_triples\\attr_triples_DBP_'+language, attr_triples)


run_dbp('ja')
