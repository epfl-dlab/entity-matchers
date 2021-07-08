import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import torch
import argparse
from tqdm import tqdm


parser = argparse.ArgumentParser(
    description="Translate the attributes of a dataset using MBart50."
)
parser.add_argument(
    "--dataset_name",
    type=str,
    choices=['ja_en', 'zh_en', 'EN_DE_15K_V1', 'EN_FR_15K_V1', 'EN_JA_15K', 'EN_DE_15K', 'EN_FR_15K'],
    help='Name of the dataset, only ja_en, zh_en, EN_DE_15K_V1, EN_FR_15K_V1, EN_JA_15K are supported at the moment'
)
parser.add_argument(
    "--attribute_file",
    type=str,
    help='Path to file with the attributes to translate.'
)
parser.add_argument(
    "--output_file",
    type=str,
    help='Path to file where translated attributes are going to be stored.'
)
parser.add_argument(
    "--src_language",
    type=str,
    help='code for the source language: (ja_XX or zh_CN etc) '
         'https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt'
)
parser.add_argument(
    "--dst_language",
    type=str,
    default='en_XX',
    help='code for the destination language: (en_XX by default) '
         'https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt'
)
parser.add_argument(
    "--device",
    type=str,
    help='device where to run the experiment, either cuda:0 either cpu'
)


def get_datasets(dataset_name: str, attr_file_folder: str) -> pd.DataFrame:
    """

    Parameters
    ----------
    dataset_name: choices are ja_en, zh_en, EN_DE_15K_V1, EN_FR_15K_V1
    attr_file_folder: path to the attribute file

    Returns
    -------
    dataframe with entity, attribute, value columns
    """
    # Create a copy of the attribute dataframe, but without discarding digits
    elems = []
    if dataset_name in ['EN_DE_15K_V1', 'EN_FR_15K_V1', 'EN_JA_15K', 'EN_FR_15K', 'EN_DE_15K']:
        with open(attr_file_folder, 'r') as f:
            for l in f:
                elems.append({
                    'entity': l.split("\t")[0],
                    'attribute': l.split("\t")[1],
                    'value': l.split("\t")[2].rstrip("\n")
                })
    elif dataset_name in ['ja_en', 'zh_en']:
        with open(attr_file_folder) as f:
            for l in f:
                line = {
                    "entity": l.split("<")[1].split(">")[0],
                    'attribute': l.split("<")[2].split(">")[0],
                    'value': " ".join(l.split(" ")[2:]).rstrip("\n")
                }
                elems.append(line)
    else:
        # At the moment, only ja_en zh_en EN_DE_15K_V1 and EN_FR_15K_V1 are supported
        raise ValueError("The only datasets supported are ja_en, zh_en, EN_DE_15K_V1, EN_FR_15K_V1")
    # df_other.loc[:, 'value'] = df_other['value'].apply(lambda x: str(x).split("^^")[0])
    # merge results with ground truth
    return elems


if __name__ == "__main__":
    args_main = parser.parse_args()
    device = args_main.device
    if device == 'cpu':
        torch.set_num_threads(4)
    # Get the model for many-to-many translations (and send it to the correct device)
    model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt").to(device)
    # Get the tokenizer, and set the correct language
    tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    tokenizer.src_lang = args_main.src_language

    rows = get_datasets(args_main.dataset_name, args_main.attribute_file)

    # Translate the text
    for i, l in tqdm(enumerate(rows)):
        text = l['value'].split('"')[1]
        encoded_text = tokenizer(text, return_tensors="pt").to(device)
        generated_tokens = model.generate(
            **encoded_text,
            forced_bos_token_id=tokenizer.lang_code_to_id[args_main.dst_language]
        )
        #df_attrs.loc[i, 'translation'] = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        l['translation'] = l['value'].replace(text, translation)
    # Write to the new attribute_file
    with open(args_main.output_file, 'w') as f:
        for i, l in tqdm(enumerate(rows)):
            print("{}\t{}\t{}".format(
                l['entity'],
                l['attribute'],
                l['translation'].rstrip(".")), file=f)
