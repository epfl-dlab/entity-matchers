import subprocess
import time
import os


def turn_yg(triples, triple_type='rel'):
    predix_dict = {'dbp': 'http://dbpedia.org/ontology/',
                   'owl': 'http://www.w3.org/2002/07/owl#',
                   'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
                   'rdfs': 'http://www.w3.org/2000/01/rdf-schema#',
                   'skos': 'http://www.w3.org/2004/02/skos/core#',
                   'xsd': 'http://www.w3.org/2001/XMLSchema#'}
    base_prefix = 'http://yago-knowledge.org/resource/'
    triples_new = set()
    for (s, p, o) in triples:
        s = s.lstrip('<').rstrip('>')
        p = p.lstrip('<').rstrip('>')
        o = o.lstrip('<').rstrip('>')
        s = base_prefix + s
        if ':' in p and 'EntityMatchers' not in p:
            p = p.split(':')
            p = predix_dict[p[0]] + p[1]
        elif 'EntityMatchers' not in p:
            p = base_prefix + p
        if triple_type == 'rel':
            o = base_prefix + o
        triples_new.add((s, p, o))
    return triples_new


def turn_and_write(rel_triples, attr_triples, seeds_triples, out_path):
    file = open(out_path, 'w', encoding='utf-8')
    for (s, p, o) in rel_triples:
        file.write('<' + s + '> <' + p + '> <' + o + '> .\n')
    for (s, p, o) in attr_triples:
        # FIX literal without quotes
        if not o.startswith('"'):
            mod_o = '"' + o + '"'
        else:
            mod_o = o
        file.write('<' + s + '> <' + p + '> ' + mod_o + ' .\n')
    for (s, p, l) in seeds_triples:
        file.write('<' + s + '> <' + p + '> ' + l + ' .\n')
    file.close()


def seed_triples(folder, dataset_division, fold_num):
    """
    Returns two lists of triples, with the seed for PARIS well formatted into
    ({resource}, {#label}, {label})
    Parameters
    ----------
    folder
    dataset_division
    fold_num

    Returns
    -------
    seed_triples_1
    seed_triples_2
    """
    root_fold = folder + dataset_division + "/" + fold_num + "/"
    seed_triples_1 = []
    seed_triples_2 = []
    with open(root_fold + "train_links") as f:
        for l in f:
            e1, e2 = l.strip("\n").split("\t")
            label = e1.split("/")[-1]
            # Use label definition for N-Triples from W3C RDF recommendation. See https://www.w3.org/TR/n-triples/
            label_str = '{resource} EntityMatchers:label "{label}"'

            # Add the label
            seed_triples_1.append((e1, "EntityMatchers:label", '"{}"'.format(label)))
            seed_triples_2.append((e2, "EntityMatchers:label", '"{}"'.format(label)))
    return seed_triples_1, seed_triples_2


def create_nt(folder, dataset_division, fold_num, kg1: str, kg2: str):
    rel_triples_1 = read_triples(folder + 'rel_triples_1')
    attr_triples_1 = read_triples(folder + 'attr_triples_1')

    if 'YG' in folder:
        rel_triples_2 = turn_yg(read_triples(folder + 'rel_triples_2'))
        attr_triples_2 = turn_yg(read_triples(folder + 'attr_triples_2'), triple_type='attr')
    else:
        rel_triples_2 = read_triples(folder + 'rel_triples_2')
        attr_triples_2 = read_triples(folder + 'attr_triples_2')

    seed_triples_1, seed_triples_2 = seed_triples(folder, dataset_division, fold_num)
    if 'YG' in folder:
        seed_triples_2 = turn_yg(seed_triples_2, 'attr')
    turn_and_write(rel_triples_1, attr_triples_1, seed_triples_1, kg1)
    turn_and_write(rel_triples_2, attr_triples_2, seed_triples_2, kg2)


def read_triples(file_path):
    """
    read relation / attribute triples from file
    :param file_path: relation / attribute triples file path
    :return: relation / attribute triples
    """
    triples = set()
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip('\n').split('\t')
            triples.add((line[0], line[1], line[2]))
    file.close()
    return triples


def run_paris(root_folder, name, ontology1, ontology2):
    current_time = time.localtime()
    ontology1 = os.path.abspath(ontology1)
    ontology2 = os.path.abspath(ontology2)

    task_name = '%s_%02d%02d_%02d%02d%02d' % (name, current_time.tm_mon, current_time.tm_mday,
                                              current_time.tm_hour, current_time.tm_min, current_time.tm_sec)

    task_name = root_folder + "/" + task_name
    os.mkdir(task_name)
    os.mkdir('%s/output' % task_name)
    os.mkdir('%s/log' % task_name)

    with open(task_name + '/paris.ini', 'w') as ini_file:
        ini_file.write('resultTSV = %s/output\n' % task_name)
        ini_file.write('factstore1 = %s\n' % ontology1)
        ini_file.write('factstore2 = %s\n' % ontology2)
        ini_file.write('home = %s/log\n' % task_name)

    _ = subprocess.call(['java', '-Xmx26000m', '-jar', 'paris.jar', task_name + '/paris.ini'])
    return task_name


def compute_prec_rec_f1(aligns, truth_links):
    """
    Note that aligns should have been pruned from the alignments already present in the
    seed.
    Truth link, hence, must contain only test and valid links.
    Parameters
    ----------
    aligns
    truth_links

    Returns
    -------

    """
    aligns = set(aligns)
    truth_links = set(truth_links)
    num_correct = len(aligns.intersection(truth_links))
    if num_correct == 0 or len(aligns) == 0:
        print("Got 0, 0, 0 in evaluation!!!")
        return 0, 0, 0
    precision = num_correct / len(aligns)
    recall = num_correct / len(truth_links)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def evaluate_paris(paris_out_folder, dataset_folder, dataset_division, fold_num):
    run = 0
    res_list = []
    while True:
        full_path = paris_out_folder + "/output/{run}_eqv.tsv".format(run=run)
        if os.path.exists(full_path):
            # PARIS create an empty file at the last_iter+1. If we encountered it, we can break
            if os.stat(full_path).st_size == 0:
                break
        run += 1
    full_path = paris_out_folder + "/output/{run}_eqv.tsv".format(run=run - 1)
    # Get PARIS result from the .tsv and elaborate it a bit to be compared with the same_list
    with open(full_path) as f:
        for l in f:
            (e1, e2, _) = l.split("\t")
            if "dbp:" in e1:
                e1 = e1.replace('dbp:', 'http://dbpedia.org/')
            if "y2:" in e2:
                e2 = e2.replace('y2:', "")
            res_list.append((e1, e2))
    fold_folder = dataset_folder + dataset_division + "/" + fold_num + "/"
    set_train = set()
    with open(fold_folder + "train_links") as f:
        for l in f:
            (e1, e2) = l.rstrip("\n").split("\t")
            set_train.add((e1, e2))
    res_no_train = []
    for align in res_list:
        if align not in set_train:
            # Evaluate only the alignments that were not present in the training data (seed).
            res_no_train.append(align)
    test_links = []
    valid_links = []
    # Create the test alignments, which are the ones contained in test and valid.
    with open(fold_folder + "test_links") as f:
        for l in f:
            (e1, e2) = l.rstrip("\n").split("\t")
            test_links.append((e1, e2))
    with open(fold_folder + "valid_links") as f:
        for l in f:
            (e1, e2) = l.rstrip("\n").split("\t")
            valid_links.append((e1, e2))
    return compute_prec_rec_f1(res_no_train, test_links + valid_links)


def parse_stats_from_log(log_file):
    """
    Read statistics of precision recall and f1-score from log files of RDGCN and BootEA
    Parameters
    ----------
    log_file name of the log file

    Returns
    -------
    precision_no_csls
    precision_csls
    recall_no_csls
    recall_csls
    f1_no_csls
    f1_csls
    """
    with open(log_file) as log:
        log_str = log.read()

    f1_no_csls = float(log_str.split("Final test result:\n\t")[-1].split("F1: ")[1].split("\n")[0])
    precision_no_csls = float(log_str.split("Final test result:\n\t")[-1].split("Precision: ")[1].split("\n")[0])
    recall_no_csls = float(log_str.split("Final test result:\n\t")[-1].split("Recall: ")[1].split("\n")[0])

    f1_csls = float(log_str.split("Final test result with csls:\n\t")[-1].split("F1: ")[1].split("\n")[0])
    precision_csls = float(log_str.split("Final test result with csls:\n\t")[-1].split("Precision: ")[1].split("\n")[0])
    recall_csls = float(log_str.split("Final test result with csls:\n\t")[-1].split("Recall: ")[1].split("\n")[0])

    train_time_seconds = float(log_str.split("Training ends. Total time = ")[-1].split(" s.")[0])
    test_time_seconds = float(log_str.split("Total run time = ")[-1].split(" s.")[0]) - train_time_seconds

    return precision_no_csls, precision_csls, recall_no_csls, recall_csls, f1_no_csls, \
           f1_csls, train_time_seconds, test_time_seconds
