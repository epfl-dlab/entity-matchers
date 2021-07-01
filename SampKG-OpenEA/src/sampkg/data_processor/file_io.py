import os
import random
import copy
from tqdm import tqdm


def read_links(file_path):
    """
    read entity links between two KGs from file
    :param file_path: entity links file path
    :return: ent_links
    """
    ent_links = set()
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in tqdm(file):
            line = line.strip('\n').split('\t')
            ent_links.add((line[0], line[1]))
    file.close()
    print('read_links:', file_path, 'ent_link_num:', len(ent_links))
    return ent_links


def write_links(file_path, ent_links):
    """
    write entity links between two KGs to file
    :param file_path: entity links output path
    :param ent_links: entity links to write
    """
    file = open(file_path, 'w', encoding='utf-8')
    for (e1, e2) in ent_links:
        output = e1 + '\t' + e2 + '\n'
        file.write(output)
    file.close()
    print('write_links:', file_path, 'ent_link_num:', len(ent_links))
    return


def read_triples(file_path):
    """
    read relation / attribute triples from file
    :param file_path: relation / attribute triples file path
    :return: relation / attribute triples
    """
    triples = set()
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in tqdm(file):
            line = line.strip('\n').split('\t')
            triples.add((line[0], line[1], line[2]))
    file.close()
    print('read_triples:', file_path, 'triple_num:', len(triples))
    return triples


def write_triples(file_path, triples):
    """
    write relation / attribute triples to file
    :param file_path: relation / attribute triples output path
    :param triples: relation / attribute triples to write
    """
    file = open(file_path, 'w', encoding='utf-8')
    output = ''
    for (e, p, o) in triples:
        output = e + '\t' + p + '\t' + o + '\n'
        file.write(output)
    file.close()
    print('write_triples:', file_path, 'triple_num:', len(triples))
    return


def split_and_write_entity_links(ent_links, output_folder, dataset_division='631'):
    test_ratio = int(dataset_division[0]) * 0.1
    train_ratio = int(dataset_division[1]) * 0.1
    valid_ratio = int(dataset_division[2]) * 0.1
    assert int(test_ratio + train_ratio + valid_ratio) == 1

    size = len(ent_links)
    ent_links = list(ent_links)
    random.shuffle(ent_links)
    test_links = set(ent_links[:int(size * test_ratio)])
    train_links = set(ent_links[int(size * test_ratio):int(size * (test_ratio + train_ratio))])
    valid_links = set(ent_links[int(size * (test_ratio + train_ratio)):])
    assert len((test_links | train_links | valid_links)) == len(ent_links)
    assert len(set(ent_links) - (test_links | train_links | valid_links)) == 0
    assert len((test_links | train_links | valid_links) - set(ent_links)) == 0

    split_links_folder = output_folder + dataset_division + '/'
    if not os.path.exists(split_links_folder):
        os.mkdir(split_links_folder)
    write_links(split_links_folder + 'test_links', test_links)
    write_links(split_links_folder + 'train_links', train_links)
    write_links(split_links_folder + 'valid_links', valid_links)
    return


def split_and_write_entity_links_5fold(ent_links, output_folder, dataset_division='721'):
    test_ratio = int(dataset_division[0]) * 0.1
    train_ratio = int(dataset_division[1]) * 0.1
    valid_ratio = int(dataset_division[2]) * 0.1
    assert int(test_ratio + train_ratio + valid_ratio) == 1

    split_links_folder = output_folder + dataset_division + '_5fold/'
    if not os.path.exists(split_links_folder):
        os.mkdir(split_links_folder)

    size = len(ent_links)
    size_train = size // 5
    ent_links_list = list(copy.deepcopy(ent_links))
    train_links_all = set()
    for i in range(1, 6):
        print('write_links', i)
        train_links = set(ent_links_list[int(size_train*(i-1)): int(size_train*i)])
        train_links_all.update(train_links)
        ent_links_test_valid = list(ent_links - train_links)
        assert len(ent_links_test_valid) == int(size*(test_ratio+valid_ratio))
        random.shuffle(ent_links_test_valid)
        test_links = set(ent_links_test_valid[:int(size * test_ratio)])
        valid_links = set(ent_links_test_valid[int(size * test_ratio):])
        assert len((test_links | train_links | valid_links)) == len(ent_links)
        assert len(set(ent_links) - (test_links | train_links | valid_links)) == 0
        assert len((test_links | train_links | valid_links) - set(ent_links)) == 0
        split_links_folder_i = split_links_folder + str(i) + '/'
        if not os.path.exists(split_links_folder_i):
            os.mkdir(split_links_folder_i)
        write_links(split_links_folder_i + 'test_links', test_links)
        write_links(split_links_folder_i + 'train_links', train_links)
        write_links(split_links_folder_i + 'valid_links', valid_links)
    assert len(ent_links - train_links_all) == len(train_links_all - ent_links) == 0
    print('\n')


if __name__ == '__main__':
    # folder = 'H:/workspace/datasets/'
    # datasets = ['EN_FR_15K_V1', 'EN_FR_15K_V2', 'EN_FR_100K_V1', 'EN_FR_100K_V2',
    #             'EN_DE_15K_V1', 'EN_DE_15K_V2', 'EN_DE_100K_V1', 'EN_DE_100K_V2',
    #             'D_W_15K_V1', 'D_W_15K_V2', 'D_W_100K_V1', 'D_W_100K_V2',
    #             'D_Y_15K_V1', 'D_Y_15K_V2', 'D_Y_100K_V1', 'D_Y_100K_V2']
    # for dataset in datasets:
    #     path = folder + dataset + '/'
    #     split_and_write_entity_links(read_links(path + 'ent_links'), path, dataset_division='811')
    folder = 'H:\workspace\VLDB2020\datasets\D_W_1M_V1\\'
    split_and_write_entity_links_5fold(read_links(folder+'ent_links'), folder)


