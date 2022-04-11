
# entity-matchers
Source code for "A Critical Re-evaluation of Neural Methods for Entity Alignment [Experiments, Analyses & Benchmarks]"

## Source code references
The source code for the various techniques has been obtained from the corresponding repositories. The code is not 100% equal due to some modifications used to run all of them under the same settings. The repositories are:
- [OpenEA](https://github.com/nju-websoft/OpenEA) for RDGCN and BootEA. The authors also provided us the source code for their IDS algorithm, that we modified into our IDS* algorithm.
- [BERT-INT](https://github.com/kosugi11037/bert-int) for BERT-INT.
- [TransEdge](https://github.com/nju-websoft/TransEdge) for TransEdge.
- [PARIS](http://webdam.inria.fr/paris/) for PARIS (you have to download the JAR).
- [DeepMatcher](https://github.com/anhaidgroup/deepmatcher) for DeepMatcher.

## Installation process
In order to run the code, you need to do the following steps: 
  1. Clone the repository using the command `git clone https://github.com/epfl-dlab/entity-matchers.git` 
  and `cd` in it: `cd entity-matchers`.
  (In case you have an ssh key set up with Github, you can clone the repository using the more convenient command 
  `git clone git@github.com:epfl-dlab/entity-matchers.git`)
  
  2. Install Anaconda (follow the instructions that you can find on the official documentation
  page https://docs.anaconda.com/anaconda/install/).
  
  3. Create a virtual environment with Anaconda, and install the dependencies 
  (we have already created a yml file with all the necessary dependencies). 
  If you do not have a GPU on your machine, then it is necessary for you change the line 151 of
  the file `entity_match.yml`, and substitute `tensorflow-gpu==1.15.0` with `tensorflow==1.15.0`. 
  After you have done, just run `conda env create --file entity_match.yml --name entity_match`. 
  If you prefer another name for the virtual environment, just change `entity_match` with your 
  desired name. This may take a while, depending on your internet connection. When it has finished,
  activate the virtual environment: `conda activate entity_match`.

  4. Install the OpenEA package: cd in `OpenEA_Mod/` and type the command `pip install -e .`. 

  5. Create an environment to run experiments with BERT-INT (you can't use the same environment as before due to version conflicts). The command we use to create such environment is `conda create --name bert-int python=3.6.9 pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 transformers=2.1.1 numpy --channel conda-forge --channel pytorch`. Note that you must match the Python, Pytorch and Transformers versions (as the ones used by the authors), otherwise BERT-INT will not work. Watch out for the cudatoolkit version and install the correct one according to your CUDA version. Note that BERT-INT can't work without a GPU so you must have CUDA installed in your system.
  6. Create an environment to run experiments with DeepMatcher. It is important that you install the package not from PyPI, but from the source code contained in the directory `/deepmatcher`. Hence, after having created the environment, cd to the directory `/deepmatcher` and type `python setup.py install`.
  7. Download the datasets: you can find them following the link [https://drive.google.com/drive/folders/1x-8OonL8SMDpNyfGyBmwzsgQL_zVMojx?usp=sharing](https://drive.google.com/drive/folders/1x-8OonL8SMDpNyfGyBmwzsgQL_zVMojx?usp=sharing). 
  Extract the zip in any directory (you will need to provide the path to the datasets later).
  
  8. Download the word embeddings at the link https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip.
  Unzip them and put them in any directory (you will need to provide the path to the .vec file as well).

## Reproduction of results
In order to run any of the experiments, cd in `/src/experiments`, activate the correct virtual environment (`bert-int` for BERT-INT experiments and `entity-match` for all the others) and run the following command:

```
python3 -u ../run_experiment.py \
        --method METHOD \
        --root_dataset DATASET_ROOT \
        --dataset DATASET \
        --dataset_division DATASET_DIVISION \
        --out_folder OUT_FOLDER \
        --gpu GPU_ID \
        --main_embeds MAIN_FILE \
        --dict_path DESC_FILE \
        --args ARGUMENTS_FILE 
        --use_func > LOG_FILE.log 
```
Where the following commands should be substituted with:
  1. `METHOD`: `RDGCN`, `BOOTEA`, `BERT-INT`, `TRANSEDGE` or `PARIS`, according to the experiment you want to replicate
  2. `DATASET_ROOT`: path to the directory that contains the dataset you need. For example, assume you want to run the main
  experiment on the DBP_en_YG_en_15K, and you have downloaded the datasets in your `~/Downloads` folder, then `DATASET_ROOT`
  should be: `~/Downloads/entity-matchers-dataset/RealEA` (note that there must be no final slash).
  3. `DATASET`: the name of the dataset, in the example above `DBP_en_YG_en_15K`.
  4. `DATASET_DIVISION`: the dataset division in folds, in our experiments `721_5folds` which stands for 70% test, 20% train, 10% validation, in 5 random folds (in order to repeat the same experiment 5 times, for robustness). The only time when you shall specify a different argument than 721_5folds is when runnign experiments with the `increasing_seed` dataset: in such case, it is enough to use as argument `631_5folds/SEED` where `SEED` is any among `1_seed`, `5_seed`, `10_seed`, `20_seed`, `30_seed`.
  5. `OUT_FOLDER`: folder where you want to store the output of the approach (and the final alignments). We recommend you create an `output/`  folder in the root directory of the repository, and for every experiment you create its own subfolder (like `output/RealEA/DBP_en_YG_en_15K` and so on).
  6. `GPU_ID`: (only for RDGCN, BootEA, BERT-INT, TransEdge) mention it only if you have more than one GPU on your machine (ids are integers starting from zero, check for them using the command `nvidia-smi`). If you are running PARIS, or if you have one or no GPU at all, do not use the argument `--gpu` in the first place. 
  7. `MAIN_FILE`: the main file for embeddings experiments: 
  - `../../OpenEA_Mod/run/main_from_args.py` for RDGCN and BootEA.
  - `../../bert-int/run_full_bert.py` for BERT-INT.
  - `../../TransEdge/code/transedge_ea.py` for TransEdge.  
  If you are running PARIS do not use the argument `--main_embeds`.
  8. `DESC_FILE`: only needed for the BERT-INT experiments with crosslingual datasets. You can find the file with the descriptions where you downloaded the datasets, under `Descriptions/BERT_INT_DESC.pkl`.
  9. `ARGUMENTS_FILE`: useful only if you are running BootEA, RDGCN, BERT-INT or TransEdge, use the correct hyper parameter file that you can find under `/src/experiments/args_best`. If you are running PARIS, do not use this argument.
  10. `use_func`: Use this flag if you want to run the modified version of BootEA that use the relation functionalities to compute the alignment loss. If you don't specifiy the flag the classic BootEA will be run
  11. `LOG_FILE.log`: file where the output will be written. At the end of the log file you will find the F1-score of the run.

Finally, note that, when running RDGCN, it is necessary to specify the location of the word embeddings file  (`wiki-news-300d-1M.vec`): in order to do so, open the directory `src/experiments/args_best`, modify the `rdgcn_args_*.json` files putting the absolute path of the word embeddings as param `word_embeds`.

Just to give an example, assuming that we want to replicate the result of the paper's Table 3 for DB-YG-15K, then the following command will do the job:
```
python3 -u ../run_experiment.py \
        --method RDGCN \
        --root_dataset ~/Downloads/entity-matchers-dataset/RealEA \
        --dataset DBP_en_YG_en_15K \
        --dataset_division 721_5folds \
        --out_folder output/RealEA/RDGCN_DBP_YG_15K \
        --gpu 0 \
        --main_embeds ../../OpenEA_Mod/run/main_from_args.py \
        --args args_best/rdgcn_args_DBP_YG_15K.json > output/RealEA/RDGCN_DBP_YG_15K/log_file.log 
``` 

To reproduce experiments of DeepMatcher, simply cd to the folder `/deepmatcher`, and launch the script `scripts/run_deepmacther.sh` with the folliwing command:

```
./scripts/run_deepmatcher.sh {logs_folder} {original_dataset_folder} {deepmatcher_dataset_folder}
```

where the arguments are:
- `{logs_folder}`: folder where you want to keep the output of the run (with precision, recall and F1-score). Notice, in such folder 5 new sub-directories will be created: `EN-DE-15K`, `EN-FR-15K`, `EN-JA-15K`, `DB-WD-15K`, `DB-YG-15K`, and in each of the sub-folders you will have the output of the 5 folds for that dataset.
- {original_dataset_folder}: folder where the dataset have been downloaded. If you downloaded them under the directory `Downloads`, then they will be under  `~/Downloads/entity-matchers-datasets`.
- {deepmatcher_dataset_folder}: folder where the dataset with deepmatcher format are kept. If you downloaded the datasets under the directory `Downloads`, they will be under the directory `~/Downloads/entity-matchers-datasets/deepmatcher`.

## Datasets description
Here is a short description of the datasets that you can find in the datasets zip:

```
├── RealEA
│   ├── DBP_en_WD_en_100K
│   ├── DBP_en_WD_en_15K
│   ├── DBP_en_YG_en_100K
│   └── DBP_en_YG_en_15K
├── AttRealEA_All
│   ├── DBP_en_WD_en_15K
│   └── DBP_en_YG_en_15K
├── AllRealEA_None
│   ├── DBP_en_WD_en_15K
│   ├── DBP_en_YG_en_15K
├── SupRealEA
│   ├── DBP_en_WD_en_15K
│   └── DBP_en_YG_en_15K
├── XRealEA
│   ├── EN_FR_15K
│   ├── EN_DE_15K
│   └── EN_JA_15K
├── XRealEA_Translated
│   ├── EN_FR_15K
│   ├── EN_DE_15K
│   └── EN_JA_15K
├── XRealEA_Pure
│   └── EN_JA_15K
├── SpaRealEA
│   └── DBP_en_YG_en_15K
└── Descriptions
    └── BERT_INT_DESC.pkl
```
- `RealEA` datasets are the ones you need to reproduce the results of Table 3 (RealEA). 
- `AttRealEA_All` are used to reproduce the results of AttRealEA of Table 3, with more attributes.
- `AttRealEA_None` are the ones you need to reproduce AttRealEA with total absence of attributes.
- `SupRealEA` are used to reproduce SupRealEA (Figure 3). Note that such datasets have a slight different structure.
- `XRealEA` is the experiment with crosslingual datasets of Table 3 (Point d).
- `XRealEA_Translated` contains the translated version of the cross-lingual datasets
- `XRealEA_Pure` is the experiment with pure crosslingual datasets of Table 3 (Point d).
- `SpaRealEA` to reproduce the experiment with the sparse dataset (present in the appendix https://www.dropbox.com/s/97g87odnxqpyz6m/EA_ReEval_full.pdf?dl=0).
- `Descriptions` contains the DBpedia abstracts used as BERT-INT descriptions (as explained in the reproduction section).

## Datasets sampling
The source code to run the IDS* algorithm is provided under `SampKG-OpenEA` folder for your convenience. If you want to create samples of any pair of KGs, you need to follow these steps (after you activated the environment):

1. Locate the directory `full_kgs` under the downloaded `entity-matchers-dataset` zip. The folder contains a directory for each datasets (containing relation triples plus attribute triples) plus a directory (`alignment`) containing the seed alignment between DBpedia and YAGO (`ent_links_dbpedia_yago`), betweehn DBpedia and Wikidata (`ent_links_dbpedia_wikidata`), and between English DBpedia and the other languages (`ent_links_en_lang`, where `lang` can be `fr`, `ja` or `de`).
2. Locate the file `strategy.py` under `SampKG-OpenEA/src/sampkg/generator` and add a new dictionary of the following format (the example reported is for the DBP_en_YG_en_100K dataset):
```
SAMPLE_NAME = {
    'max_degree_kg1': 100,
    'max_degree_kg2': 100,
    'delete_ratio': 0.03,
    'delete_random_ratio': 0.2,
    'delete_degree_ratio': 0.15,
    'delete_limit': 1.1,
    'preserve_num': 500
}
```
Where the parameters must be changed in the following way:
- `SAMPLE_NAME`: is the name to give to the sample (note this name must be used later on when running the script).
- `max_degree_kg1`: maximum entity degree that will be considered during sampling for KG1 (all the entities in KG1 with degree > `max_degree_kg1` will be considered as having degree = `max_degree_kg1`)
- `max_degree_kg2`: same as `max_degree_kg1` for KG2.
- `delete_ratio`: the ratio of entities that will be deleted at each step of IDS* (Parameter $\mu$ from the original algorithm). The higher this value, more the entities deleted at each step and faster the algorithm (Note that if you choose a too high value the algorithm may fail, deleting more entities than the given limit)
- `delete_random_ratio`: the ratio of entities randomly deleted at each step (NOTE: used only if the entities deleted by PageRank probabilistically are not enough for the given step)
- `delete_degree_ratio`: when `delete_limit` is reached entities start to be deleted by degree (from the lowest to the highest) up to when the correct number of entities is reached (as deleting only by PageRank is too much unpredictable). This parameter measure the ratio of entities deleted this way at every step.
- `delete_limit`: when to stop deleting using PageRank and start deleting by degree with respect to the number of matchable entities you want.
- `preserve_num`: number of entities with highest degree to preserve. If this number is greater than 0, `preserve_num` entities with the highest degree will be added again and IDS* will delete by degree up to when the number of matchable entities required is reached again.

Hyper-parameters settings to produce datasets similar to the one we used are already given inside `strategy.py`.

3. Move to the `SampKG-OpenEA/sampkg/src` folder and run the following command:
```
  python3 main.py \
    --target_dataset SAMPLE_NAME \
    --KG1_rel_triple_path PATH_REL_TRIPLES_1 \
    --KG1_attr_triple_path PATH_ATTR_TRIPLES_1 \
    --KG2_rel_triple_path PATH_REL_TRIPLES_2 \
    --KG2_attr_triple_path PATH_ATTR_TRIPLES_2 \
    --ent_link_path PATH_LINK \
    --ent_link_num NUM_MATCH \
    --delete_param DELETE_PARAM \
    --output_folder OUT_FOLDER
```
Where the parameters are:
- `SAMPLE_NAME`: is the name to give to the sample (must be the same as the one in `strategy.py`
- `KG1_rel_triple_path`: path to the relation triples of the KG1.
- `KG1_attr_triple_path`: path to the attribute triples of the KG1.
- `KG2_rel_triple_path`: path to the relation triples of the KG2.
- `KG2_attr_triple_path`: path to the attribute triples of the KG2.
- `ent_link_path`: path to the ground truth for the pair of KGs.
- `ent_link_num`: number specifying how many matchable entities there will be in the final sample
- `delete_param`: the upscale factor presented in the algorithm IDS* from the paper. The higher the factor, smaller the size of non-matchable entities.
- `output_folder`: folder where the sample will be stored (under `OUT_FOLDER/SAMPLE_NAME`). Note that you must give a `/` at the end of the folder path

Finally, note that relation triples and attribute triples must follow the same format used among all of our datasets, i.e. `ENTITY TAB RELATION/PROPERTY TAB ENTITY/LITERAL` and the ground truth must be instead of the form `ENTITY1 TAB ENTITY2`(as it is for the datasets we provide).

For example, a dataset similar to our DBP_en_YG_en_100K can be obtained with the command (supposing that the `full_kgs` folder is copied inside `SampKG-OpenEA/sampkg/src` and a folder `output/` exists in the same directory). A `delete_param` of 2.5 will produce around 30% of non-matchable entities for this dataset.
```
  python3 main.py \
      --target_dataset DBP_en_YG_en_100K \
      --KG1_rel_triple_path dbpedia/rel_triples \
      --KG1_attr_triple_path dbpedia/attr_triples \
      --KG2_rel_triple_path yago3.1/rel_triples \
      --KG2_attr_triple_path yago3.1/attr_triples \
      --ent_link_path alignment/ent_links_dbpedia_yago \
      --ent_link_num 100000 \
      --delete_param 2.5 \
      --pre_delete 0 \
      --output_folder output/
```
