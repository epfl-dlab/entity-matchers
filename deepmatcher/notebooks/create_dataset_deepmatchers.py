import deepmatcher as dm
# Import py_entitymatching package
import py_entitymatching as em
import os
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(
    description="Create datasets compatible with deepmatchers"
)

parser.add_argument(
    "--input_dataset_folder",
    type=str,
    help=''
)
parser.add_argument(
    "--output_dataset_folder",
    type=str,
    help=''
)
parser.add_argument(
    "--threshold_names",
    type=float,
    help="threshold for names jaccard similarity. The lower the threshold the higher the number of rows kept."
)
parser.add_argument(
    "--threshold_other_attributes",
    type=float,
    help="threshold for other_attributes jaccard similarity. The lower the threshold the higher the number of rows kept."
)
parser.add_argument(
    "--threshold_one_hop_names",
    type=float,
    help="threshold for 1 hop names jaccard similarity. The lower the threshold the higher the number of rows kept."
)
parser.add_argument(
    "--threshold_one_hop_other_attributes",
    type=float,
    help="threshold for 1 hop other_attributes jaccard similarity. The lower the threshold the higher the number of rows kept."
)
parser.add_argument(
    "--threshold_relations",
    type=float,
    help="threshold for relations jaccard similarity. The lower the threshold the higher the number of rows kept."
)

args_main = parser.parse_args()

candidate_attributes = {
    "DB-WD-15K": [
        [
            'http://dbpedia.org/ontology/abbreviation', 'http://dbpedia.org/ontology/alias', 'http://dbpedia.org/ontology/birthName',
            'http://dbpedia.org/ontology/formerName', 'http://dbpedia.org/ontology/longName', 'http://dbpedia.org/ontology/name', 'http://dbpedia.org/ontology/otherName',
            'http://dbpedia.org/ontology/synonym', 'http://dbpedia.org/ontology/teamName', 'http://dbpedia.org/ontology/title', 'http://xmlns.com/foaf/0.1/givenName',
            'http://xmlns.com/foaf/0.1/name', 'http://xmlns.com/foaf/0.1/nick'
        ],
        [
            'http://www.wikidata.org/prop/direct/P2611', 'http://www.wikidata.org/prop/direct/P1813', 'http://www.wikidata.org/prop/direct/P742',
            'http://www.wikidata.org/prop/direct/P1477', 'http://www.wikidata.org/prop/direct/P1448', 'http://www.wikidata.org/prop/direct/P1705',
            'http://www.wikidata.org/prop/direct/P2003', 'http://www.wikidata.org/prop/direct/P2002', 'http://www.wikidata.org/prop/direct/P1549',
            'http://www.w3.org/2004/02/skos/core#altLabel', 'http://www.wikidata.org/prop/direct/P1559', 'http://www.wikidata.org/prop/direct/P1472',
            'http://www.wikidata.org/prop/direct/P1476', 'http://www.wikidata.org/prop/direct/P1449', 'http://www.wikidata.org/prop/direct/P428',
        ]
    ],
    "DB-YG-15K":[
        [
            'http://dbpedia.org/ontology/pseudonym', 'http://dbpedia.org/ontology/name',
            'http://xmlns.com/foaf/0.1/name', 'http://xmlns.com/foaf/0.1/givenName',
            'http://dbpedia.org/ontology/longName', 'http://dbpedia.org/ontology/birthName',
            'http://dbpedia.org/ontology/formerName'
        ],
        [
            'rdfs:label', 'skos:prefLabel', 'hasFamilyName', 'hasGivenName'
        ]
    ],
    "DB-WD-100K": [
        [
            'http://dbpedia.org/ontology/abbreviation', 'http://dbpedia.org/ontology/alias', 'http://dbpedia.org/ontology/birthName',
            'http://dbpedia.org/ontology/formerName', 'http://dbpedia.org/ontology/longName', 'http://dbpedia.org/ontology/name', 'http://dbpedia.org/ontology/otherName',
            'http://dbpedia.org/ontology/synonym', 'http://dbpedia.org/ontology/teamName', 'http://dbpedia.org/ontology/title', 'http://xmlns.com/foaf/0.1/givenName',
            'http://xmlns.com/foaf/0.1/name', 'http://xmlns.com/foaf/0.1/nick'
        ],
        [
            'http://www.wikidata.org/prop/direct/P2611', 'http://www.wikidata.org/prop/direct/P1813', 'http://www.wikidata.org/prop/direct/P742',
            'http://www.wikidata.org/prop/direct/P1477', 'http://www.wikidata.org/prop/direct/P1448', 'http://www.wikidata.org/prop/direct/P1705',
            'http://www.wikidata.org/prop/direct/P2003', 'http://www.wikidata.org/prop/direct/P2002', 'http://www.wikidata.org/prop/direct/P1549',
            'http://www.w3.org/2004/02/skos/core#altLabel', 'http://www.wikidata.org/prop/direct/P1559', 'http://www.wikidata.org/prop/direct/P1472',
            'http://www.wikidata.org/prop/direct/P1476', 'http://www.wikidata.org/prop/direct/P1449', 'http://www.wikidata.org/prop/direct/P428',
            'http://www.wikidata.org/prop/direct/P373', "http://schema.org/name", "http://www.w3.org/2004/02/skos/core#prefLabel",
            "http://www.w3.org/2000/01/rdf-schema#label"
        ]
    ],
    "DB-YG-100K":[
        [
            'http://dbpedia.org/ontology/pseudonym', 'http://dbpedia.org/ontology/name',
            'http://xmlns.com/foaf/0.1/name', 'http://xmlns.com/foaf/0.1/givenName',
            'http://dbpedia.org/ontology/longName', 'http://dbpedia.org/ontology/birthName',
            'http://dbpedia.org/ontology/formerName'
        ],
        [
            'rdfs:label', 'skos:prefLabel', 'hasFamilyName', 'hasGivenName', 'redirectedFrom'
        ]
    ],
    "EN-DE-15K": [
        [
            'alias', 'birthName', 'formerName', 'http://xmlns.com/foaf/0.1/givenName', 'longName',
            'http://xmlns.com/foaf/0.1/name', 'http://xmlns.com/foaf/0.1/nick', 'otherName', 'pseudonym',
            'synonym', 'teamName'
        ],
        [
            'alternativeName', 'birthName', 'colourName', 'formerName', 'historicalName',
            'http://xmlns.com/foaf/0.1/name', 'http://xmlns.com/foaf/0.1/nick', 'scientificName'
        ]
    ],
    "EN-FR-15K": [
        [
            'alias', 'birthName', 'formerName', 'http://xmlns.com/foaf/0.1/givenName',
            'longName', 'http://xmlns.com/foaf/0.1/name', 'http://xmlns.com/foaf/0.1/nick', 'otherName', 'pseudonym',
            'synonym', 'teamName'
        ],
        [
            'alternativeName', 'birthName', 'colonialName', 'formerName', 'http://xmlns.com/foaf/0.1/name',
            'http://xmlns.com/foaf/0.1/nick', 'officialName', 'oldName', 'originalName', 'otherName',
            'peopleName', 'personName', 'policeName', 'spouseName'
         ]
    ],
    "EN-JA-15K":[
        [
            'alias', 'birthName', 'formerName', 'http://xmlns.com/foaf/0.1/givenName', 'longName',
            'http://xmlns.com/foaf/0.1/name', 'http://xmlns.com/foaf/0.1/nick', 'otherName', 'pseudonym', 'synonym'
        ],
        [
            'alias', 'birthName', 'commonName', 'formerName', 'http://xmlns.com/foaf/0.1/givenName', 'longName',
            'http://xmlns.com/foaf/0.1/name', 'http://xmlns.com/foaf/0.1/nick', 'pseudonym',
            'http://xmlns.com/foaf/0.1/surname', 'synonym'
        ]
    ]
}


def get_input(dataset):
    # First, we need to recognize which are the entity names/aliases
    with open("{}/attr_triples_1".format(dataset)) as f:
        triples_att_1 = []
        for l in f:
            e, a, v = l.rstrip().split("\t")
            triples_att_1.append((e, a, v))
    with open("{}/attr_triples_2".format(dataset)) as f:
        triples_att_2 = []
        for l in f:
            e, a, v = l.rstrip().split("\t")
            triples_att_2.append((e, a, v))
    return triples_att_1, triples_att_2


def has_numbers(s):
    # Might be a bit harsh!
    return any(char.isdigit() for char in s)


def create_df_strategy_1(dataset_folder, dataset_number, candidate_names):
    candidates_names_set = set(candidate_names)  # Make a set for faster lookup
    entity_values = {} # Indexed by entity name.
    # Scan attributes
    with open("{}/attr_triples_{}".format(dataset_folder, str(dataset_number))) as f:
        attr_triples_by_entity = {}
        for l in f:
            e, a, v = l.rstrip("\n").split("\t")
            if e not in attr_triples_by_entity:
                attr_triples_by_entity[e] = []
            v2 = re.sub(r'@([a-z]+)', "", v)
            attr_triples_by_entity[e].append((e, a, v2))
            if e not in entity_values:
                entity_values[e] = {"names": set(), "other_attributes": set(), "1-hop-names": set(), "1-hop-other_attributes": set()}
    print("Scanned attributes")
    # Scan relations
    with open("{}/rel_triples_{}".format(dataset_folder, str(dataset_number))) as f:
        rel_by_entity = {}
        rel_names_by_entity = {}
        for l in f:
            e1, r, e2 = l.rstrip("\n").split("\t")
            if e1 not in rel_by_entity:
                rel_by_entity[e1] = []
            if e1 not in rel_names_by_entity:
                rel_names_by_entity[e1] = set()
            rel_by_entity[e1].append(e2)
            if e2 not in rel_by_entity:
                rel_by_entity[e2] = []
            if e2 not in rel_names_by_entity:
                rel_names_by_entity[e2] = set()
            rel_by_entity[e2].append(e1)
            rel_names_by_entity[e1].add(r.split("/")[-1])
            rel_names_by_entity[e2].add(r.split("/")[-1])
            if e1 not in entity_values:
                entity_values[e1] = {"names": set(),
                                     "other_attributes": set(),
                                     "relations": set(),
                                     "1-hop-names": set(),
                                     "1-hop-other_attributes": set()}
            if e2 not in entity_values:
                entity_values[e2] = {"names": set(),
                                     "other_attributes": set(),
                                     "relations": set(),
                                     "1-hop-names": set(),
                                     "1-hop-other_attributes": set()}
    print("Scanned relations")
    # Add names and other attributes
    for e in entity_values:
        if e in attr_triples_by_entity:
            for e2, a, v in attr_triples_by_entity[e]:
                if not has_numbers(v):
                    if a in candidates_names_set:  # Remove numbers!
                        # Remove all " characters and commas.
                        entity_values[e]['names'].add(" ".join(v.replace('"', '').split(",")))
                    else:
                        entity_values[e]['other_attributes'].add(" ".join(v.replace('"', '').split(",")))

    print("Made names and ohter attributes")
    # Add 1-hop names and other attributes
    for e in tqdm(entity_values):
        for neigh in rel_by_entity[e]:
            if neigh in attr_triples_by_entity:
                for neight2, a, v in attr_triples_by_entity[neigh]:
                    if not has_numbers(v):
                        if a in candidates_names_set:  # Remove numbers!
                            entity_values[e]['1-hop-names'].add(" ".join(v.replace('"', '').split(",")))
                        else:
                            entity_values[e]['1-hop-other_attributes'].add(" ".join(v.replace('"', '').split(",")))
    print("Made one-hop names and other attributes")
    for e in entity_values:
        entity_values[e]['names'] = " ".join(list(entity_values[e]['names']))
        entity_values[e]['other_attributes'] = " ".join(list(entity_values[e]['other_attributes']))
        entity_values[e]['1-hop-names'] = " ".join(list(entity_values[e]['1-hop-names']))
        entity_values[e]['1-hop-other_attributes'] = " ".join(list(entity_values[e]['1-hop-other_attributes']))
        entity_values[e]['relations'] = " ".join(list(rel_names_by_entity[e]))
    return entity_values


def perform_blocking(table_name, dataset_folder, col_name, threshold):
    print("Blocking by {}".format(col_name))
    # Perform blocking
    path_A = "{}/{}A.csv".format(dataset_folder, table_name)
    path_B = "{}/{}B.csv".format(dataset_folder, table_name)
    # Read the CSV files and set 'ID' as the key attribute
    A = em.read_csv_metadata(path_A, key='id')
    B = em.read_csv_metadata(path_B, key='id')
    print(len(A), len(B))
    block_f = em.get_features_for_blocking(A, B, validate_inferred_attr_types=False)
    rb = em.RuleBasedBlocker()
    try:
        # Add rule : block tuples if name_name_lev(ltuple, rtuple) < 0.4
        rb.add_rule(['{}_{}_jac_qgm_3_qgm_3(ltuple, rtuple) < {}'.format(col_name, col_name, threshold)], block_f)
        return rb.block_tables(A, B, l_output_attrs=['id', col_name], r_output_attrs=['id', col_name], show_progress=True)
    except (AttributeError, AssertionError):
        print("it was not possible to block by {}".format(col_name))
        return None


def save_dataframes(df_train, df_test, df_valid, columns, out_folder):
    columns_expanded = []
    for x in columns:
        columns_expanded.append("ltable_{}".format(x))
        columns_expanded.append("rtable_{}".format(x))
    columns_expanded.append("id")
    columns_expanded.append("label")
    df_train[columns_expanded].to_csv("{}/train.csv".format(out_folder), index=False)
    df_test[columns_expanded].to_csv("{}/test.csv".format(out_folder), index=False)
    df_valid[columns_expanded].to_csv("{}/valid.csv".format(out_folder), index=False)


datasets_supported = ['DB-WD-15K', 'DB-YG-15K', 'DB-WD-100K', 'DB-YG-100K', 'EN-DE-15K', 'EN-FR-15K', 'EN-JA-15K']


if __name__ == "__main__":
    triples_att_1, triples_att_2 = get_input(args_main.input_dataset_folder)
    dataset_name = ""
    for d in datasets_supported:
        if d in args_main.input_dataset_folder:
            dataset_name = d
            break
    assert dataset_name != ""
    candidates_attributes_1 = candidate_attributes[dataset_name][0]
    candidates_attributes_2 = candidate_attributes[dataset_name][1]

    ent_values_1 = create_df_strategy_1(args_main.input_dataset_folder, 1, candidates_attributes_1)
    ent_values_2 = create_df_strategy_1(args_main.input_dataset_folder, 2, candidates_attributes_2)

    # Create dataframes
    df_a = pd.DataFrame([{"id": e,
                          "names": ent_values_1[e]['names'],
                          "other_attributes": ent_values_1[e]['other_attributes'],
                          "one_hop_names": ent_values_1[e]['1-hop-names'],
                          "one_hop_other_attributes": ent_values_1[e]['1-hop-other_attributes'],
                          "relations": ent_values_1[e]['relations']} for e in ent_values_1],
                          )
    df_b = pd.DataFrame([{"id": e,
                          "names": ent_values_2[e]['names'],
                          "other_attributes": ent_values_2[e]['other_attributes'],
                          "one_hop_names": ent_values_2[e]['1-hop-names'],
                          "one_hop_other_attributes": ent_values_2[e]['1-hop-other_attributes'],
                          "relations": ent_values_2[e]['relations']} for e in ent_values_2])

    df_a.to_csv("{}/tableA.csv".format(args_main.output_dataset_folder))
    df_b.to_csv("{}/tableB.csv".format(args_main.output_dataset_folder))

    # First no emtpy names (They must have length >= 3)
    df_a[df_a.names.str.len() >= 7].to_csv("{}/tableNamesA.csv".format(args_main.output_dataset_folder), index=False)
    df_b[df_b.names.str.len() >= 7].to_csv("{}/tableNamesB.csv".format(args_main.output_dataset_folder), index=False)

    # First no emtpy other attributes
    df_a[df_a.other_attributes.str.len() >= 3].to_csv("{}/tableOtherAttrA.csv".format(args_main.output_dataset_folder), index=False)
    df_b[df_b.other_attributes.str.len() >= 3].to_csv("{}/tableOtherAttrB.csv".format(args_main.output_dataset_folder), index=False)

    # First no emtpy one_hop_nanes
    df_a[df_a.one_hop_names.str.len() >= 3].to_csv("{}/tableOneHopNamesA.csv".format(args_main.output_dataset_folder), index=False)
    df_b[df_b.one_hop_names.str.len() >= 3].to_csv("{}/tableOneHopNamesB.csv".format(args_main.output_dataset_folder), index=False)

    # First no emtpy relations
    # Remove slashes.
    df_a['relations'] = df_a['relations'].apply(lambda x: " ".join(y.split("/")[-1] if "/" in y else y for y in x.split(" ")))
    df_b['relations'] = df_b['relations'].apply(lambda x: " ".join(y.split("/")[-1] if "/" in y else y for y in x.split(" ")))
    df_a[df_a.relations.str.len() >= 3].to_csv("{}/tableRelationsA.csv".format(args_main.output_dataset_folder),
                                                   index=False)
    df_b[df_b.relations.str.len() >= 3].to_csv("{}/tableRelationsB.csv".format(args_main.output_dataset_folder),
                                                   index=False)

    # First no emtpy other attributes
    df_a[df_a.one_hop_other_attributes.str.len() >= 3].to_csv("{}/tableOneHopOtherAttrA.csv".format(args_main.output_dataset_folder),
                                                              index=False)
    df_b[df_b.one_hop_other_attributes.str.len() >= 3].to_csv("{}/tableOneHopOtherAttrB.csv".format(args_main.output_dataset_folder),
                                                              index=False)

    C_names = perform_blocking("tableNames", args_main.output_dataset_folder, 'names', args_main.threshold_names)
    C_other_attributes = perform_blocking("tableOtherAttr", args_main.output_dataset_folder, 'other_attributes', args_main.threshold_other_attributes)
    C_one_hop_names = perform_blocking("tableOneHopNames", args_main.output_dataset_folder, 'one_hop_names', args_main.threshold_one_hop_names)
    C_one_hop_other_attributes = perform_blocking("tableOneHopOtherAttr", args_main.output_dataset_folder, 'one_hop_other_attributes', args_main.threshold_one_hop_other_attributes)
    C_relations = perform_blocking("tableRelations", args_main.output_dataset_folder,
                                                  'relations',
                                                  args_main.threshold_relations)

    df_a.columns = ["ltable_" + x for x in df_a.columns]
    df_b.columns = ["rtable_" + x for x in df_b.columns]

    list_dfs = []
    if C_names is not None:
        df_names = C_names.copy()
        df_names_merged = df_names.merge(df_a,
                                         left_on=['ltable_id', 'ltable_names'],
                                         right_on=['ltable_id', 'ltable_names']
                                         ).merge(df_b,
                                                 left_on=['rtable_id', 'rtable_names'],
                                                 right_on=['rtable_id', 'rtable_names'])
        list_dfs.append(df_names_merged)
        print("Blocked by names:", len(df_names_merged))
    if C_one_hop_names is not None:
        df_one_hop_names = C_one_hop_names.copy()
        df_one_hop_names_merged = df_one_hop_names.merge(df_a,
                                              left_on=['ltable_id', 'ltable_one_hop_names'],
                                              right_on=['ltable_id', 'ltable_one_hop_names']
                                              ).merge(df_b,
                                                      left_on=['rtable_id', 'rtable_one_hop_names'],
                                                      right_on=['rtable_id', 'rtable_one_hop_names'])
        list_dfs.append(df_one_hop_names_merged)
        print("Blocked by one_hop_names:", len(df_one_hop_names_merged))
    if C_other_attributes is not None:
        df_other_attributes = C_other_attributes.copy()
        df_other_attributes_merged = df_other_attributes.merge(df_a,
                                                               left_on=['ltable_id', 'ltable_other_attributes'],
                                                               right_on=['ltable_id', 'ltable_other_attributes']
                                                               ).merge(df_b,
                                                                       left_on=['rtable_id', 'rtable_other_attributes'],
                                                                       right_on=['rtable_id',
                                                                                 'rtable_other_attributes'])
        list_dfs.append(df_other_attributes_merged)
        print("Blocked by other_attributes:", len(df_other_attributes_merged))
    if C_one_hop_other_attributes is not None:
        df_one_hop_other_attributes = C_one_hop_other_attributes.copy()
        df_one_hop_other_attributes_merged = df_one_hop_other_attributes.merge(df_a,
                                                                    left_on=['ltable_id',
                                                                             'ltable_one_hop_other_attributes'],
                                                                    right_on=['ltable_id',
                                                                              'ltable_one_hop_other_attributes']
                                                                    ).merge(df_b,
                                                                            left_on=['rtable_id',
                                                                                     'rtable_one_hop_other_attributes'],
                                                                            right_on=['rtable_id',
                                                                                      'rtable_one_hop_other_attributes'])
        list_dfs.append(df_one_hop_other_attributes_merged)
        print("Blocked by 1 hop other attributes:", len(df_one_hop_other_attributes_merged))
    if C_relations is not None:
        df_relations = C_relations.copy()
        df_relations_merged = df_relations.merge(df_a,
                                                left_on=['ltable_id',
                                                         'ltable_relations'],
                                                right_on=['ltable_id',
                                                          'ltable_relations']
                                                ).merge(df_b,
                                                        left_on=['rtable_id',
                                                                 'rtable_relations'],
                                                        right_on=['rtable_id',
                                                                  'rtable_relations'])
        list_dfs.append(df_relations_merged)
        print("Blocked by relations:", len(df_relations_merged))

    df_concat = pd.concat(list_dfs)
    print("Rows of blocked dataset {}".format(len(df_concat)))
    df_concat = df_concat.drop(columns=['_id'])
    df_concat_copy = df_concat.copy()
    # Remove duplicates:
    df_concat_no_duplicates = df_concat.drop_duplicates(['ltable_id', 'rtable_id'])
    print("Rows of blocked dataset {}".format(len(df_concat_no_duplicates)))

    df_a_final = df_a.copy()
    df_a_final.columns = [x.replace("ltable_", "") for x in df_a_final.columns]
    df_a_final.to_csv("{}/tableA.csv".format(args_main.output_dataset_folder))

    df_b_final = df_b.copy()
    df_b_final.columns = [x.replace("rtable_", "") for x in df_b_final.columns]
    df_b_final.to_csv("{}/tableB.csv".format(args_main.output_dataset_folder))

    for fold in range(1, 6):  # Do the 5 folds

        # Add rows for the ground truth of train and valid.
        ent_links1 = {}
        ent_links2 = {}
        with open("{}/721_5folds/{}/train_links".format(args_main.input_dataset_folder, fold)) as f:
            for l in f:
                e1, e2 = l.rstrip("\n").split("\t")
                assert e1 not in ent_links1 and e2 not in ent_links2
                ent_links1[e1] = e2
                ent_links2[e2] = e1
        with open("{}/721_5folds/{}/valid_links".format(args_main.input_dataset_folder, fold)) as f:
            for l in f:
                e1, e2 = l.rstrip("\n").split("\t")
                assert e1 not in ent_links1 and e2 not in ent_links2
                ent_links1[e1] = e2
                ent_links2[e2] = e1

        print("size of ground truth", len(ent_links1), len(ent_links2))
        set_covered_links = set()
        for i, r in df_concat_no_duplicates.iterrows():
            if r['ltable_id'] in ent_links1 and ent_links1[r['ltable_id']] == r['rtable_id']:
                set_covered_links.add(r['ltable_id'])

        new_rows = []
        for e1 in tqdm(ent_links1):
            if e1 not in set_covered_links:
                e1_row = df_a[df_a.ltable_id == e1].iloc[0]
                e2_row = df_b[df_b.rtable_id == ent_links1[e1]].iloc[0]
                new_rows.append(
                    {
                        "ltable_id": e1_row.ltable_id,
                        "ltable_names": e1_row.ltable_names,
                        "ltable_other_attributes": e1_row.ltable_other_attributes,
                        "ltable_one_hop_names": e1_row.ltable_one_hop_names,
                        "ltable_one_hop_other_attributes": e1_row.ltable_one_hop_other_attributes,
                        "ltable_relations": e1_row.ltable_relations,
                        "rtable_id": e2_row.rtable_id,
                        "rtable_names": e2_row.rtable_names,
                        "rtable_other_attributes": e2_row.rtable_other_attributes,
                        "rtable_one_hop_names": e2_row.rtable_one_hop_names,
                        "rtable_one_hop_other_attributes": e2_row.rtable_one_hop_other_attributes,
                        "rtable_relations": e2_row.rtable_relations,
                    })

        df_missing_rows = pd.DataFrame(new_rows)

        df_blocked_pairs = df_concat_no_duplicates.append(df_missing_rows, ignore_index=True)
        # Append labels
        labels = []
        gt_1_to_2 = {}
        with open("{}/ent_links".format(args_main.input_dataset_folder)) as f:
            for l in f:
                e1, e2 = l.rstrip("\n").split("\t")
                gt_1_to_2[e1] = e2
        for i, r in df_blocked_pairs.iterrows():
            if r['ltable_id'] in gt_1_to_2 and gt_1_to_2[r['ltable_id']] == r['rtable_id']:
                labels.append(1)
            else:
                labels.append(0)
        df_blocked_pairs['label'] = labels
        # Count how many labels 1 there are: We want 15k of course!!
        print("Number of pairs in GT (they may not be 15K):", df_blocked_pairs['label'].sum())

        # working command
        # python create_dataset_deepmatchers.py --input_dataset_folder /home/stefano/Documents/git/EPFL/dlab/datasets/entity-matchers-dataset/RealEA/DB-YG-15K --output_dataset_folder RealEA/DB-YG-15K --threshold_names 0.4 --threshold_other_attributes 0.4 --threshold_one_hop_names 0.4 --threshold_one_hop_other_attributes 0.4

        # Take 3 samples of label = 0 of our blocked pairs df and re-distribute them among train, test and validation.
        df_wrong_blocked_pairs = df_blocked_pairs[df_blocked_pairs['label'] == 0]

        df_test = []
        df_train = []
        df_valid = []

        # For the pairs in the test, keep only the ones that the blocking algo could detect correctly.
        with open("{}/721_5folds/{}/test_links".format(args_main.input_dataset_folder, fold)) as f:
            df_test_all = []
            for l in f:
                e1, e2 = l.rstrip("\n").split("\t")
                df_test_all.append({"ltable_id": e1, "rtable_id": e2})
            for i, l in df_blocked_pairs[df_blocked_pairs['label'] == 1].merge(pd.DataFrame(df_test_all),
                                                                   left_on=['ltable_id', 'rtable_id'],
                                                                   right_on=['ltable_id', 'rtable_id']).iterrows():
                df_test.append({"ltable_id": l.ltable_id, "rtable_id": l.rtable_id})

        with open("{}/721_5folds/{}/train_links".format(args_main.input_dataset_folder, fold)) as f:
            for l in f:
                e1, e2 = l.rstrip("\n").split("\t")
                df_train.append({"ltable_id": e1, "rtable_id": e2})

        with open("{}/721_5folds/{}/valid_links".format(args_main.input_dataset_folder, fold)) as f:
            for l in f:
                e1, e2 = l.rstrip("\n").split("\t")
                df_valid.append({"ltable_id": e1, "rtable_id": e2})

        print("len of datasets:", len(df_valid), len(df_train), len(df_test), len(df_wrong_blocked_pairs))
        df_shuffled = df_wrong_blocked_pairs.sample(frac=1.0)  # Shuffle.

        # 70% test, 20% train, 10% valid
        for i, l in df_shuffled.iterrows():
            if i < 0.7 * len(df_shuffled):
                df_test.append({"ltable_id": l.ltable_id, "rtable_id": l.rtable_id})
            elif i < 0.9 * len(df_shuffled):
                df_train.append({"ltable_id": l.ltable_id, "rtable_id": l.rtable_id})
            else:
                df_valid.append({"ltable_id": l.ltable_id, "rtable_id": l.rtable_id})

        df_train_merged = pd.DataFrame(df_train).merge(df_blocked_pairs, left_on=['ltable_id', 'rtable_id'],
                                                       right_on=['ltable_id', 'rtable_id'])
        df_test_merged = pd.DataFrame(df_test).merge(df_blocked_pairs, left_on=['ltable_id', 'rtable_id'],
                                                     right_on=['ltable_id', 'rtable_id'])
        df_valid_merged = pd.DataFrame(df_valid).merge(df_blocked_pairs, left_on=['ltable_id', 'rtable_id'],
                                                       right_on=['ltable_id', 'rtable_id'])
        df_train_merged['id'] = list(range(0, len(df_train_merged)))
        df_test_merged['id'] = list(range(len(df_train_merged), len(df_test_merged) + len(df_train_merged)))
        df_valid_merged['id'] = list(range(len(df_test_merged) + len(df_train_merged),
                                           len(df_test_merged) + len(df_train_merged) + len(df_valid_merged)))
        df_train_merged.head()

        try:
            os.makedirs("{}/721_5folds/{}".format(args_main.output_dataset_folder, fold))

        except OSError:
            print("Creation of the directories failed")
        else:
            print("Successfully created the directory")

        save_dataframes(df_train_merged, df_test_merged, df_valid_merged,
                        ['names', 'other_attributes', 'one_hop_names', 'one_hop_other_attributes', 'id', 'relations'],
                        "{}/721_5folds/{}".format(args_main.output_dataset_folder, fold))

    # To delete the tableNamesA.csv, tableNamesB.csv and so on.:
    to_delete_files = ['tableNames', 'tableOneHopNames', 'tableOtherAttr', 'tableOneHopOtherAttr', 'tableRelations']
    for to_delete_file in to_delete_files:
        try:
            os.remove("{}/{}A.csv".format(args_main.output_dataset_folder, to_delete_file))
            os.remove("{}/{}B.csv".format(args_main.output_dataset_folder, to_delete_file))
        except OSError:
            print("It was not possible to delete the table...A,B.csv files")