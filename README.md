
# entity-matchers
Source code for "A Critical Re-evaluation of Neural Methods for Entity Alignment"

## Installation process
In order to run the code, you need to do the following steps: 
  1. Clone the repository using the command `git clone https://github.com/epfl-dlab/entity-matchers.git` 
  and `cd` in it: `cd entity-matchers`.
  (In case you have an ssh key set up with Github, you can clone the repository using the more convenient command 
  `git clone git@github.com:blind-anonymous/entity-matchers.git`)
  
  2. Install Anaconda (follow the instructions that you can find on the official documentation
  page https://docs.anaconda.com/anaconda/install/).
  
  3. Create a virtual environment with Anaconda, and install the dependencies 
  (we have already created a yml file with all the necessary dependencies). 
  If you do not have a GPU on your machine, then it is necessary for you change the line 151 of
  the file `env.yml`, and substitute `tensorflow-gpu==1.15.0` with `tensorflow==1.15.0`. 
  After you have done, just run `conda env create --file env.yml --name entity_match`. 
  If you prefer another name for the virtual environment, just change `entity_match` with your 
  desired name. This may take a while, depending on your internet connection. When it has finished,
  activate the virtual environment: `conda activate entity_match`.

  4. Install the openEA package: cd in `OpenEA_Mod/` and type the command `pip install -e .`. 
  
  5. Download the datasets: you can find them following the link 
  [https://drive.google.com/drive/folders/1x-8OonL8SMDpNyfGyBmwzsgQL_zVMojx?usp=sharing](https://drive.google.com/drive/folders/1x-8OonL8SMDpNyfGyBmwzsgQL_zVMojx?usp=sharing). 
  Extract the zip in any directory (you will need to provide the path to the datasets later).
  
  6. Download the word embeddings at the link https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip.
  Unzip them and put them in any directory (you will need to provide the path to the .vec file as well).

## Reproduction of results
In order to run any of the experiments, cd in `/src`, activate the previously created virtual environment and run the following command:

```
python3 -u run_experiment.py \
        --method METHOD \
        --root_dataset DATASET_ROOT \
        --dataset DATASET \
        --dataset_division DATASET_DIVISION \
        --out_folder OUT_FOLDER \
        --gpu GPU_ID \
        --main_embeds MAIN_FILE \
        --args ARGUMENTS_FILE > LOG_FILE.log 
```
Where the following commands should be substituted with:
  1. `METHOD`: `RDGCN`, `BOOTEA` or `PARIS`, according to the experiment you want to replicate
  2. `DATASET_ROOT`: path to the directory that contains the dataset you need. For example, assume you want to run the main
  experiment on the DBP_en_YG_en_15K_V1, and you have downloaded the datasets in your `~/Download` folder, then `DATASET_ROOT`
  should be: `~/Downloads/datasets/main` (note that there must be no final slash).
  3. `DATASET`: the name of the dataset, in the example above `DBP_en_YG_en_15K_V1`.
  4. `DATASET_DIVISION`: the dataset division in folds, in our experiments `721_5folds` which stands for 70% test, 20% train, 10% validation, in 5 random folds (in order to repeat the same experiment 5 times, for robustness). The only time when you shall specify a different argument than 721_5folds is when runnign experiments with the `increasing_seed` dataset: in such case, it is enough to use as argument `631_5folds/SEED` where `SEED` is any among `1_seed`, `5_seed`, `10_seed`, `20_seed`, `30_seed`.
  5. `OUT_FOLDER`: folder where you want to store the output of the approach (and the final alignments). We recommend you create an `output/`  folder in the root directory of the repository, and for every experiment you create its own subfolder (like `output/main/DBP_en_YG_en_15K_V1` and so on).
  6. `GPU_ID`: (only for RDGCN and BootEA) mention it only if you have more than one GPU on your machine (ids are integers starting from zero, check for them using the command `nvidia-smi`). If you are running PARIS, or if you have one or no GPU at all, do not use the argument `--gpu` in the first place. 
  7. `MAIN_FILE`: the main file, which is `../../OpenEA_Mod/run/main_from_args.py`. Use only for RDGCN and BootEA, if you are running PARIS do not use the argument `--main_embeds`.
  8. `ARGUMENTS_FILE`: useful only if you are running BootEA or RDGCN, use the correct hyper parameter file that you can find under `/src/experiments/`. If you are running PARIS, do not use this argument.
  9. `LOG_FILE.log`: file where the output will be written. At the end of the log file you will find the F1-score of the run.

Finally, note that, when running RDGCN, it is necessary to specify the location of the word embeddings file  (`wiki-news-300d-1M.vec`): in order to do so, open the directory `src/experiments/args_best`, modify the `rdgcn_args_*.json` files putting the absolute path of the word embeddings as param `word_embeds`.

Just to give an example, assuming that we want to replicate the result of the paper's Table 5 for DB-YG-15K, then the following command will do the job:
```
python3 -u run_experiment.py \
        --method RDGCN \
        --root_dataset ~/Downloads/datasets/main \
        --dataset DBP_en_YG_en_15K_V1 \
        --dataset_division 721_5folds \
        --out_folder output/main/RDGCN_DBP_YG_15K \
        --gpu 0 \
        --main_embeds ../OpenEA_Mod/run/main_from_args.py \
        --args experiments/args_best/rdgcn_args_DBP_YG_15K.json > output/main/RDGCN_DBP_YG_15K/log_file.log 
``` 

## Datasets description
Here is a short description of the datasets that you can find in the datasets zip:

```
├── main
│   ├── DBP_en_WD_en_100K_V1
│   ├── DBP_en_WD_en_15K_V1
│   ├── DBP_en_YG_en_100K_V1
│   └── DBP_en_YG_en_15K_V1
├── no_attributes
│   ├── DBP_en_WD_en_15K_V1_no_attr
│   └── DBP_en_YG_en_15K_V1_no_attr
├── no_extra_attributes
│   ├── DBP_en_WD_en_100K_no_extra
│   ├── DBP_en_WD_en_15K_no_extra
│   ├── DBP_en_YG_en_100K_no_extra
│   └── DBP_en_YG_en_15K_no_extra
├── increasing_seed
│   ├── DBP_en_WD_en_15K_V1_incr_seed
│   └── DBP_en_YG_en_15K_V1_incr_seed
└── sparse
    └── DBP_en_YG_en_15K_V1_sparse
```
-  `main` datasets are the ones you need to reproduce the results of Table 5 (RealEA). 
- `no_attributes` are the ones you need to reproduce AttRealEA point 1, total absence of attributes.
- `no_extra_attributes` are used to reproduce the results of AttRealEA point 2.
- `increasing_seed` are used to reproduce SupRealEA. Note that such datasets have a slight different structure.
- `sparse` is the experiment SpaRealEA.

## Datasets sampling
The source code to run the IDS* algorithm is provided under `SampKG-OpenEA` folder for your convenience. If you want to create samples of any pair of KGs, you need to follow these steps (after you activated the environment):

1. Download the full datasets zip `full_kgs.zip` from [https://zenodo.org/record/4540561#.YClg3nVKhhE	](https://zenodo.org/record/4540561#.YClg3nVKhhE) and uncompress it in a folder or your convenience. The zip contains a directory for each datasets (containing relation triples plus attribute triples) plus a directory containing the seed alignment between DBPedia and YAGO (`ent_links_dbpedia_yago`) and betweehn DBPedia and Wikidata (`ent_links_dbpedia_wikidata`).
2. Locate the file `strategy.py` under `SampKG-OpenEA/src/sampkg/generator` and add a new dictionary of the following format (the example reported is for the DBP_en_YG_en_100K_V1 dataset):
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

Finally, note that relation triples and attribute triples must follow the same format usef among all of our datasets, i.e. `ENTITY TAB RELATION/PROPERTY TAB ENTITY/LITERAL` and the ground truth must be instead of the form `ENTITY1 TAB ENTITY2`(as it is for the datasets we provide).

For example, a dataset similar to our DBP_en_YG_en_100K_V1 can be obtained with the command (supposing that the `full_kgs.zip` is decompressed inside `SampKG-OpenEA/sampkg/src` and a folder `output/` exists in the same directory). A `delete_param` of 2.5 will produce around 30% of non-matchable entities for this dataset.
```
  python3 main.py \
      --target_dataset DBP_en_YG_en_100K_V1 \
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
