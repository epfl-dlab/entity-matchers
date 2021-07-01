import pandas as pd
import numpy as np
import os, shutil


def create_label(sample_row):
    """
    Create a label for each dataset, given a SameAs row

    Args:
        sample_row (Pandas row): Row from SameAs, with the two entities which are the same

    Returns:
        sample_row (Pandas row): Same row as input with two new labeled entries
    """
    # Build label from DB (we use as label what's after the resource/)
    label = sample_row["KG1"].split("/")[-1].replace(">", "")
    # Use label definition for N-Triples from W3C RDF recommendation. See https://www.w3.org/TR/n-triples/
    label_str = '{resource} <http://www.w3.org/2000/01/rdf-schema#label> "{label}" . \n'

    # Add the label
    sample_row["KG1_labeled"] = label_str.format(resource=sample_row["KG1"], label=label)
    sample_row["KG2_labeled"] = label_str.format(resource=sample_row["KG2"], label=label)
    return sample_row


def create_sample(same_as, frac: float):
    """
    Get a sample from same_as.

    Args:
        same_as (Pandas Dataframe): Dataframe containing the ground truth
        frac (float): seed percentage to use

    Returns:
        same_as (Pandas Dataframe): Sampled dataframe with two labeled column (one for DB and one for FB)
    """
    # Get a sample and build label
    sampled = same_as.sample(
        frac=frac, replace=False
    )  # Create a sample without replacement.
    sampled = sampled.apply(lambda x: create_label(x), axis=1)
    return sampled


def load_kgs(training_data: str, dataset_type: str, triples_type: str, dataset_ea: str = None): 
    """Load Knowledge graphs

    Args:
        training_data (str): training data folder
        dataset_type (str): dataset source type
        triples_type (str): relation, attributes or full tuples
        dataset_ea (str, optional): used only for OpenEA datasets. Defaults to None.
    """

    # Get full folder structure and ground truth depending on the dataset we are using
    if dataset_type == "OpenEA_dataset":
        if dataset_ea == None:
            raise Exception("Need to specify the OpenEA dataset to use")
        folder = training_data + dataset_type + "/" + dataset_ea + "/"
    else:
        folder = training_data + dataset_type + "/"
    if triples_type == "relation":
        filename = "rel_triples"
    elif triples_type == "attribute":
        filename = "attr_triples"
    else:
        filename = "full_triples"
        build_full_triples(folder)
    
    triples1_file = open(folder + filename + "_1")
    triples2_file = open(folder + filename + "_2")
    triples1 = triples1_file.readlines()
    triples2 = triples2_file.readlines()
    if triples_type == "attribute":
        for i, t1 in enumerate(triples1):
            (p,r,o) = t1.rstrip().split("\t")
            # FIX literal without quotes
            if(not o.startswith('"')):
                mod_o = '"' + o + '"'
            else:
                mod_o = o
            triples1[i] = "\t".join([p,r,mod_o]) + "\n"
        for i, t2 in enumerate(triples2):
            (p,r,o) = t2.rstrip().split("\t")
            if(not o.startswith('"')):
                mod_o = '"' + o + '"'
            else:
                mod_o = o
            triples2[i] = "\t".join([p,r,mod_o]) + "\n"
    triples1_file.close()
    triples2_file.close()

    return triples1, triples2, folder

def add_point(triples1: list, triples2: list, folder: str):

    for i, t1 in enumerate(triples1):
            triples1[i] = t1.rstrip() + " .\n"
    for i, t2 in enumerate(triples2):
        triples2[i] = t2.rstrip() + " .\n"

    kg1_path = folder + "1_pointed.nt"
    kg2_path = folder + "2_pointed.nt"
    
    kg1_label = open(kg1_path, "w")
    kg2_label = open(kg2_path, "w")
    kg1_label.writelines(triples1)
    kg2_label.writelines(triples2)
    kg1_label.close()
    kg2_label.close()

    return kg1_path, kg2_path


def seed_kgs(dataset_type: str, triples1: list, triples2: list, folder: str, new_seed: bool, dataset_division: str = "721_5fold", fold_num: str = None, frac: float = 0):
    """[summary]

    Args:
        dataset_type (str): dataset source type
        triples1 (list): [description]
        triples2 (list): [description]
        folder (str): [description]
        new_seed (bool): [description]
        dataset_divison (str, optional): used only for OpenEA datasets. Defaults to "721_5fold".
        fold_num (str, optional): used only for OpenEA datasets. Defaults to None.
        frac (float, optional): [description]. Defaults to 0.
    """
    if dataset_type == "OpenEA_dataset":
        if not new_seed:
            seed_pd = pd.read_csv(folder + dataset_division + "/" + fold_num + "/train_links", "\t", header=None)
            seed_pd.rename(columns={0: "KG1", 1: "KG2"}, inplace=True)
            seed_pd = seed_pd.apply(lambda x: create_label(x), axis=1)
            kg1_path = folder + dataset_division + "/" + fold_num + "/1_seeded.nt"
            kg2_path = folder + dataset_division + "/" + fold_num + "/2_seeded.nt"
        else:
            same_as = pd.read_csv(folder + "ent_links", "\t", header=None)
            same_as.rename(columns={0: "KG1", 1: "KG2"}, inplace=True)
            seed_pd = create_sample(same_as, frac)
            kg1_path = folder + "/1_seeded.nt"
            kg2_path = folder + "/2_seeded.nt"

        seed1 = seed_pd["KG1_labeled"].to_list()
        seed2 = seed_pd["KG2_labeled"].to_list()

        # Add a . at the end of each line
        for i, t1 in enumerate(triples1):
            triples1[i] = t1.rstrip() + " .\n"
        for i, t2 in enumerate(triples2):
            triples2[i] = t2.rstrip() + " .\n"
        
        triples1_labeled = triples1 + seed1
        triples2_labeled = triples2 + seed2
        
        kg1_label = open(kg1_path, "w")
        kg2_label = open(kg2_path, "w")
        kg1_label.writelines(triples1_labeled)
        kg2_label.writelines(triples2_labeled)
        kg1_label.close()
        kg2_label.close()
    else:
        same_as = pd.read_csv(folder + "ent_links", " ", header=None)[[0,2]]
        same_as.rename(columns={0: "KG1", 2: "KG2"}, inplace=True)
        seed_pd = create_sample(same_as, frac)

        seed1 = seed_pd["KG1_labeled"].to_list()
        seed2 = seed_pd["KG2_labeled"].to_list()
        
        triples1_labeled = triples1 + seed1
        triples2_labeled = triples2 + seed2

        kg1_path = folder + "1_seeded.nt"
        kg2_path = folder + "2_seeded.nt"
        
        kg1_label = open(kg1_path, "w")
        kg2_label = open(kg2_path, "w")
        kg1_label.writelines(triples1_labeled)
        kg2_label.writelines(triples2_labeled)
        kg1_label.close()
        kg2_label.close()
    
    return kg1_path, kg2_path



def build_full_triples(folder: str):
    """Build a dataset of full triples from the attributes and relations

    Args:
        folder (str): folder where the files are located
    """
    attr_triples_1 = open(folder + "attr_triples_1")
    attr_triples_2 = open(folder + "attr_triples_2")
    rel_triples_1 = open(folder + "rel_triples_1")
    rel_triples_2 = open(folder + "rel_triples_2")
    attr1_lines = attr_triples_1.readlines()
    attr2_lines = attr_triples_2.readlines()
    for i, t1 in enumerate(attr1_lines):
        (p,r,o) = t1.rstrip().split("\t")
        # FIX literal without quotes
        if(not o.startswith('"')):
            mod_o = '"' + o + '"'
        else:
            mod_o = o
        attr1_lines[i] = "\t".join([p,r,mod_o]) + "\n"
    for i, t2 in enumerate(attr2_lines):
        (p,r,o) = t2.rstrip().split("\t")
        if(not o.startswith('"')):
            mod_o = '"' + o + '"'
        else:
            mod_o = o
        attr2_lines[i] = "\t".join([p,r,mod_o]) + "\n"
    rel1_lines = rel_triples_1.readlines()
    rel2_lines = rel_triples_2.readlines()
    full1 = attr1_lines + rel1_lines
    full2 = attr2_lines + rel2_lines
    full1_file = open(folder + "full_triples_1", "w")
    full2_file = open(folder + "full_triples_2", "w")
    full1_file.writelines(full1)
    full2_file.writelines(full2)
    
    attr_triples_1.close()
    attr_triples_2.close()
    rel_triples_1.close()
    rel_triples_2.close()
    return

def clean(folder:str, dataset_type: str, out_path: str, dataset_division: str = "721_5fold", fold_num: str = None):
    os.system("rm run_*")
    if dataset_type == "OpenEA_dataset":
        if fold_num == None:
            if os.path.exists(folder + "/1_pointed.nt"):
                os.remove(folder + "/1_pointed.nt")
                os.remove(folder + "/2_pointed.nt")
            else:
                os.remove(folder + "/1_seeded.nt")
                os.remove(folder + "/2_seeded.nt")
        else:
            os.remove(folder + dataset_division + "/" + fold_num + "/1_seeded.nt")
            os.remove(folder + dataset_division + "/" + fold_num + "/2_seeded.nt")
    else:
        os.remove(folder + "1_seeded.nt")
        os.remove(folder + "2_seeded.nt")
    # shutil.rmtree(out_path) will be cleaned only be a new PARIS run
