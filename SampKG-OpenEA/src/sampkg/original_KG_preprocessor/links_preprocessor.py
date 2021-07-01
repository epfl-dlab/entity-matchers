# import ftfy

from data_processor.file_io import *


def read_links_raw(file_path, language):
    prefix = '<http://'
    if language == 'en':
        prefix += 'dbpedia.org'
    else:
        prefix += language
    links = set()
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip(' .\n').split(' ')
            if len(line) != 3 or not line[2].startswith(prefix):
                continue
            line[0] = line[0].lstrip('<').rstrip('>')
            line[2] = line[2].lstrip('<').rstrip('>')
            links.add((line[0], line[2]))
    file.close()
    if language == 'en':
        links = set([(e2, e1) for (e1, e2) in links])
    return links


def read_links_raw_wd(file_path):
    links = set()
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip(' .\n').split(' ')
            if len(line) != 3:
                continue
            e1 = line[0].lstrip('<').rstrip('>')
            e2 = line[2].lstrip('<').rstrip('>')
            if 'http://www.wikidata.org/entity/Q' not in e2:
                continue
            links.add((e1, e2))
    file.close()
    return links


def read_links_raw_yg(file_path):
    links = set()
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip(' .\n').split('\t')
            if len(line) != 3 or 'http://dbpedia.org' not in line[2]:
                continue
            e1 = line[0].lstrip('<').rstrip('>')
            e2 = line[2].lstrip('<').rstrip('>')
            links.add((e2, e1))
    file.close()
    return links


def read_links_raw_geonames(file_path):
    links = set()
    ents1, ents2 = set(), set()
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip(' .\n').split('> <')
            if len(line) != 3:
                continue
            e1 = line[0].lstrip('<').rstrip('>')
            e2 = line[2].lstrip('<').rstrip('>')
            print((e1, e2))
            if e1 not in ents1 and e2 not in ents2:
                links.add((e1, e2))
                ents1.add(e1)
                ents2.add(e2)
    file.close()
    return links


# def read_links_raw_lmdb_1(file_path):
#     links = set()
#     interlink_uri_dict = {}
#     with open(file_path, 'r', encoding='utf-8') as file:
#         for line in file:
#             line = line.strip('.\n').split('> ')
#             s = line[0].lstrip('<')
#             p = line[1].lstrip('<')
#             o = line[2].lstrip('<')
#             if p == 'http://data.linkedmdb.org/resource/oddlinker/link_target' and 'dbpedia.org/resource' in o:
#                 # print(o, s)
#                 links.add((o, s))
#             elif p == 'http://data.linkedmdb.org/resource/oddlinker/link_source':
#                 interlink_uri_dict[s] = o
#     file.close()
#     links_new = set()
#     for (e1, e2) in links:
#         e1 = e1.replace('%28', '(').replace('%29', ')').replace('%21', '!').replace('%26', '&').replace('%27', "'").replace('%2C', ',')
#         e1 = ftfy.ftfy(e1)
#         links_new.add((e1, interlink_uri_dict[e2]))
#         print(e1, interlink_uri_dict[e2])
#     print(len(links_new))
#     return links_new


def read_links_raw_lmdb_2(file_path):
    links = set()
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip(' .\n').split('> <')
            if len(line) != 3:
                continue
            e1 = line[0].lstrip('<').rstrip('>')
            e2 = line[2].lstrip('<').rstrip('>')
            print((e1, e2))
            links.add((e1, e2))
    file.close()
    return links


# def run_links_lmdb():
#     ls1 = read_links_raw_lmdb_1('/media/sl/Data/workspace/VLDB/SampKG/raw_data/linkedmdb-latest-dump.nt')
#     ls2 = read_links_raw_lmdb_2('/media/sl/Data/workspace/VLDB/SampKG/raw_data/DBpedia/linkedmdb_links.nt')
#     print(len(ls1), len(ls2), len(ls1 | ls2))
#     ls = ls1 | ls2
#     ents1, ents2 = set(), set()
#     links = set()
#     for (e1, e2) in ls:
#         if e1 not in ents1 and e2 not in ents2:
#             links.add((e1, e2))
#             ents1.add(e1)
#             ents2.add(e2)
#     write_links('/media/sl/Data/workspace/VLDB/SampKG/processed_data/ent_links/ent_links_DBP_en_LMDB_en', links)


def filter_links(links, triples1, triples2, is_rel_triples=True):
    ents1 = set([h for (h, _, _) in triples1])
    ents2 = set([h for (h, _, _) in triples2])
    if is_rel_triples:
        ents1 = ents1 | set([t for (_, _, t) in triples1])
        ents2 = ents2 | set([t for (_, _, t) in triples2])
    links_new = set([(e1, e2) for (e1, e2) in links if e1 in ents1])
    links_new = set([(e1, e2) for (e1, e2) in links_new if e2 in ents2])
    return links_new


def run(path, language):
    if language == 'wd':
        en_other = read_links_raw_wd(path + 'raw_data/DBpedia/interlanguage_links_en.ttl')
        links_new = en_other
        other = 'WD_en'
    elif language == 'yg':
        other_en = read_links_raw_yg(path+'raw_data/YAGO/yagoDBpediaInstances.ttl')
        links_new = other_en
        other = 'YG_en'
    else:
        en_other = read_links_raw(path + 'raw_data/DBpedia/interlanguage_links_en.ttl', language)
        other_en = read_links_raw(path + 'raw_data/DBpedia/interlanguage_links_' + language + '.ttl', 'en')
        links_new = en_other | other_en
        other = 'DBP_' + language
    print('raw links:', len(links_new))

    rel_triples1 = read_triples(path+'processed_data/rel_triples/rel_triples_DBP_en')
    attr_triples1 = read_triples(path+'processed_data/attr_triples/attr_triples_DBP_en')
    rel_triples2 = read_triples(path + 'processed_data/rel_triples/rel_triples_'+other)
    attr_triples2 = read_triples(path + 'processed_data/attr_triples/attr_triples_'+other)

    links_new = filter_links(links_new, rel_triples1, rel_triples2)
    print('after filter_links by rel_triples:', len(links_new))
    links_new = filter_links(links_new, attr_triples1, attr_triples2, is_rel_triples=False)
    print('after filter_links by attr_triples:', len(links_new))

    # delete one2more
    one2one = {}
    for (e1, e2) in links_new:
        one2one[e1] = e2
    links_new = set([(e1, e2) for e1, e2 in one2one.items()])
    one2one = {}
    for (e1, e2) in links_new:
        one2one[e2] = e1
    links_new = set([(e1, e2) for e2, e1 in one2one.items()])
    print('after delete one2more:', len(links_new))

    write_links(path+'processed_data/ent_links/ent_links_DBP_en_'+other, links_new)


run('H:/workspace/VLDB2020/SampKG/', 'ja')
# ls = read_links_raw_geonames('/media/sl/Data/workspace/VLDB/SampKG/raw_data/DBpedia/geonames_links_en.ttl')
# write_links('/media/sl/Data/workspace/VLDB/SampKG/processed_data/ent_links/ent_links_DBP_en_GN_en', ls)

# run_links_lmdb()


